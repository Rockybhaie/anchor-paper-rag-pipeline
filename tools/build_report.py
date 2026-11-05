"""Build reports/milestone1_report.pdf from the report text and the saved results.

Tables, charts and example answers are filled in from results/, so they always match the runs.
Needs tools/requirements.txt and `playwright install chromium`.
Run from the repository root: python -m tools.build_report
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path

import markdown
from playwright.sync_api import sync_playwright

from src.evaluate import load_rows, most_changed_question

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"
TEMPLATE = REPORTS / "milestone1_report.md"
OUTPUT = REPORTS / "milestone1_report.pdf"

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
MUTED_MARK = "#a3a29c"
CHOSEN_MARK = "#2a78d6"
AXIS_MIN, AXIS_MAX = 0.85, 1.0
AXIS_TICKS = (0.85, 0.90, 0.95, 1.0)
EXCERPT_CHARS = 250

SETTING_TITLES = {
    "chunking": "Chunking (Nested Learning)",
    "embedding": "Embeddings (Nested Learning)",
    "retrieval_mode": "Retrieval (Titans)",
    "passages_used": "Passages used (Titans)",
    "prompt": "Prompts (TTT RNNs)",
    "model": "Models (Hierarchical Reasoning)",
    "temperature": "Temperature (Hierarchical Reasoning)",
}
CSS = f"""
@page {{ size: A4; margin: 17mm 16mm; }}
* {{ box-sizing: border-box; }}
body {{ font: 10.2pt/1.42 Arial, Helvetica, sans-serif; color: {INK}; margin: 0; }}
h1 {{ font-size: 19pt; margin: 0 0 2pt; }}
h2 {{ font-size: 12.5pt; margin: 14pt 0 4pt; break-after: avoid; }}
h3 {{ font-size: 10.5pt; margin: 9pt 0 3pt; break-after: avoid; }}
p {{ margin: 0 0 6pt; }} ul {{ margin: 0 0 6pt; padding-left: 16pt; }} li {{ margin-bottom: 2pt; }}
.byline {{ color: {INK_SECONDARY}; margin-bottom: 10pt; }}
table {{ border-collapse: collapse; width: 100%; margin: 4pt 0 8pt; font-size: 8.8pt;
        break-inside: avoid; }}
th {{ text-align: left; border-bottom: 1px solid {INK}; padding: 2.5pt 5pt 2.5pt 0; }}
td {{ border-bottom: 1px solid {GRID}; padding: 2.5pt 5pt 2.5pt 0; vertical-align: top; }}
td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
tr.served td {{ font-weight: bold; }}
code {{ font: 8.8pt Consolas, monospace; }}
.charts {{ display: flex; flex-wrap: wrap; gap: 6pt 14pt; margin: 4pt 0 6pt; }}
.chart {{ width: 48%; break-inside: avoid; }}
.chart h4 {{ font-size: 9pt; margin: 0 0 2pt; }}
.note {{ color: {INK_SECONDARY}; font-size: 8.6pt; }}
.example {{ break-inside: avoid; margin-bottom: 8pt; }}
.answer {{ border-left: 2px solid {GRID}; padding-left: 7pt; margin: 3pt 0; font-size: 8.8pt; }}
.answer b {{ display: block; }}
figure {{ margin: 6pt 0 9pt; break-inside: avoid; }}
figure img {{ max-width: 100%; max-height: 58mm; border: 1px solid {GRID}; display: block; }}
figcaption {{ font-size: 8.6pt; color: {INK_SECONDARY}; margin-top: 2pt; }}
.flow {{ display: flex; align-items: stretch; gap: 3pt; margin: 4pt 0 8pt; break-inside: avoid; }}
.flow div {{ flex: 1; border: 1px solid {MUTED_MARK}; border-radius: 3px; padding: 4pt 5pt;
            font-size: 8.2pt; line-height: 1.25; }}
.flow b {{ display: block; }}
.flow span {{ align-self: center; color: {INK_SECONDARY}; }}
"""


def x_position(value: float, left: float, width: float) -> float:
    clipped = min(max(value, AXIS_MIN), AXIS_MAX)
    return left + (clipped - AXIS_MIN) / (AXIS_MAX - AXIS_MIN) * width


def setting_chart(runs: list[dict]) -> str:
    """Draw one setting as a dot chart. The chosen value is blue, bold and labelled."""
    rows = sorted(runs, key=lambda run: run["metrics"]["faithfulness"], reverse=True)
    label_width, plot_left, plot_width, row_height = 118, 122, 92, 16
    width, top = 262, 4
    height = top + row_height * len(rows) + 15
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="100%" role="img" font-family="Arial, sans-serif" font-size="7.6">'
    ]
    for tick in AXIS_TICKS:
        x = x_position(tick, plot_left, plot_width)
        parts.append(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{top}" y2="{height - 12}" stroke="{GRID}"/>'
            f'<text x="{x:.1f}" y="{height - 3}" text-anchor="middle" '
            f'fill="{INK_SECONDARY}">{tick:.2f}</text>'
        )
    for position, run in enumerate(rows):
        y = top + row_height * position + row_height / 2
        value = run["metrics"]["faithfulness"]
        chosen = run["winner"]
        name = html.escape(run["variant"]) + (" (chosen)" if chosen else "")
        weight = "bold" if chosen else "normal"
        parts.append(
            f'<text x="{label_width}" y="{y + 2.6:.1f}" text-anchor="end" fill="{INK}" '
            f'font-weight="{weight}">{name}</text>'
            f'<circle cx="{x_position(value, plot_left, plot_width):.1f}" cy="{y:.1f}" r="4" '
            f'fill="{CHOSEN_MARK if chosen else MUTED_MARK}"/>'
            f'<text x="{width - 2}" y="{y + 2.6:.1f}" text-anchor="end" fill="{INK}" '
            f'font-weight="{weight}">{value:.3f}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def stage_charts(index: list[dict]) -> str:
    panels = []
    for setting, title in SETTING_TITLES.items():
        runs = [run for run in index if run["setting"] == setting]
        panels.append(f'<div class="chart"><h4>{title}</h4>{setting_chart(runs)}</div>')
    return (
        f'<div class="charts">{"".join(panels)}</div>'
        '<p class="note">Faithfulness on 15 questions per chart. '
        "The axis starts at 0.85 to show small differences.</p>"
    )


def table(headers: list[str], rows: list[list[str]], numeric_from: int, served: int = -1) -> str:
    def cell(tag: str, column: int, value: str) -> str:
        kind = ' class="num"' if column >= numeric_from else ""
        return f"<{tag}{kind}>{value}</{tag}>"

    head = "".join(cell("th", i, value) for i, value in enumerate(headers))
    body = []
    for number, row in enumerate(rows):
        mark = ' class="served"' if number == served else ""
        cells = "".join(cell("td", i, value) for i, value in enumerate(row))
        body.append(f"<tr{mark}>{cells}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def judge_table(index: list[dict]) -> str:
    runs = [run for run in index if run["setting"] == "judge"]
    runs.sort(key=lambda run: run["metrics"]["judge_gap"], reverse=True)
    rows = [
        [
            html.escape(run["variant"]) + (" (chosen)" if run["winner"] else ""),
            f"{run['metrics']['faithfulness']:.3f}",
            f"{run['metrics']['wrong_context_faithfulness']:.3f}",
            f"{run['metrics']['judge_gap']:.3f}",
        ]
        for run in runs
    ]
    headers = ["Judge", "On real passages", "On unrelated passages", "Gap"]
    return table(headers, rows, numeric_from=1, served=0)


def final_table(best_run: dict) -> str:
    runs = sorted(best_run["all"], key=lambda r: r["metrics"]["faithfulness"], reverse=True)
    rows, served = [], -1
    for number, run in enumerate(runs):
        metrics = run["metrics"]
        config = run["config"]
        name = run["name"].split("-", 1)[0]
        if run["name"] == best_run["serving"]["name"]:
            served = number
            name += " (served)"
        if run["name"] == best_run["best"]["name"]:
            name += " (top score)"
        rows.append(
            [
                name,
                config["retrieval"]["mode"].replace("_", " + "),
                config["generation"]["model"],
                f"{metrics['faithfulness']:.3f}",
                f"{metrics['relevancy']:.3f}",
                f"{metrics['relevancy_when_answered']:.3f}",
                f"{metrics['source_hit_rate']:.2f}",
                f"{metrics['refusal_rate']:.0%}",
                f"{metrics['total_ms'] / 1000:.1f} s",
            ]
        )
    headers = ["Setup", "Retrieval", "Generator", "Faithful.", "Relevancy", "Rel. if answered"]
    headers += ["Right paper", "Refusals", "Avg time"]
    return table(headers, rows, numeric_from=3, served=served)


def anchor_table(index: list[dict], best_run: dict) -> str:
    anchors = ("nested_learning", "titans", "ttt_rnn", "hrm")
    runs = sorted(best_run["all"], key=lambda r: r["metrics"]["faithfulness"], reverse=True)
    rows = []
    for run in runs:
        scores = {
            item["anchor"]: item["metrics"]["faithfulness"]
            for item in index
            if item["stage"] == "final" and item["variant"] == run["name"]
        }
        row = [run["name"].split("-", 1)[0]] + [f"{scores[anchor]:.3f}" for anchor in anchors]
        rows.append(row)
    headers = ["Setup", "Nested Learning", "Titans", "TTT RNNs", "Hierarchical Reasoning"]
    return table(headers, rows, numeric_from=1)


def corpus_table() -> str:
    names = {
        "nested_learning": "Nested Learning: The Illusion of Deep Learning Architectures",
        "titans": "Titans: Learning to Memorize at Test Time",
        "ttt_rnn": "Learning to (Learn at Test Time): RNNs with Expressive Hidden States",
        "hrm": "Hierarchical Reasoning Model",
    }
    rows = []
    for anchor, title in names.items():
        papers = json.loads((ROOT / f"data/anchors/{anchor}/manifest.json").read_text("utf-8"))
        total = len(papers["papers"])
        pages = sum(paper["pages"] for paper in papers["papers"])
        rows.append([title, str(total), str(pages)])
    return table(["Anchor paper", "Papers in corpus", "Pages"], rows, numeric_from=1)


def example(index: list[dict], setting: str) -> str:
    """Show the question this setting changed most: the chosen run and the most different one."""
    runs = [run for run in index if run["setting"] == setting]
    answers = [load_rows(RESULTS / run["answers_path"]) for run in runs]
    scores = [load_rows(RESULTS / run["scores_path"]) for run in runs]
    question_id = most_changed_question(answers, scores)
    rows = list(zip(runs, answers, scores, strict=True))

    def score_of(item: tuple) -> float:
        value = item[2][question_id].get("faithfulness")
        return -1.0 if value is None else value

    chosen = next(item for item in rows if item[0]["winner"])
    others = [item for item in rows if item is not chosen]
    different = max(others, key=lambda item: abs(score_of(item) - score_of(chosen)))
    parts = [
        f'<div class="example"><p><b>{SETTING_TITLES[setting]}.</b> '
        f"{html.escape(answers[0][question_id]['question'])}</p>"
    ]
    for item, label in ((chosen, "chosen"), (different, "most different")):
        run, answer_rows, score_rows = item
        faithfulness = score_rows[question_id].get("faithfulness")
        shown = "refused" if faithfulness is None else f"faithfulness {faithfulness:.2f}"
        text = " ".join(answer_rows[question_id]["answer"].split())
        if len(text) > EXCERPT_CHARS:
            text = text[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " ..."
        parts.append(
            f'<div class="answer"><b>{html.escape(run["variant"])} ({label}, {shown})</b>'
            f"{html.escape(text)}</div>"
        )
    return "".join(parts) + "</div>"


def screenshot(name: str, caption: str) -> str:
    uri = (REPORTS / "figures" / f"{name}.png").as_uri()
    return f'<figure><img src="{uri}" alt="{caption}"><figcaption>{caption}</figcaption></figure>'


def fill_tokens(text: str, index: list[dict], best_run: dict) -> str:
    text = text.replace("{{stage_charts}}", stage_charts(index))
    text = text.replace("{{judge_table}}", judge_table(index))
    text = text.replace("{{final_table}}", final_table(best_run))
    text = text.replace("{{anchor_table}}", anchor_table(index, best_run))
    text = text.replace("{{corpus_table}}", corpus_table())
    text = re.sub(r"\{\{example:(\w+)\}\}", lambda m: example(index, m.group(1)), text)
    return re.sub(
        r"\{\{shot:(\w+)\|(.+?)\}\}", lambda m: screenshot(m.group(1), m.group(2)), text
    )


def main() -> None:
    index = json.loads((RESULTS / "runs_index.json").read_text("utf-8"))
    best_run = json.loads((RESULTS / "best_run.json").read_text("utf-8"))
    source = fill_tokens(TEMPLATE.read_text("utf-8"), index, best_run)
    body = markdown.markdown(source, extensions=["tables", "md_in_html"])
    page_html = '<!doctype html><meta charset="utf-8"><title>Milestone 1</title>'
    page_html += f"<style>{CSS}</style>{body}"
    temporary = REPORTS / "_report.html"
    temporary.write_text(page_html, "utf-8")
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="chromium")
            page = browser.new_page()
            page.goto(temporary.as_uri())
            page.pdf(path=OUTPUT, format="A4", print_background=True, prefer_css_page_size=True)
            browser.close()
    finally:
        temporary.unlink()
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
