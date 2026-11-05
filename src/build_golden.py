"""Create the question set for each anchor. Run once, then review and commit the result.

Questions are drawn from across every paper, and about one in five compares two papers. The earlier
project drew them only from each paper's first four passages. Each anchor gets 15 questions for
choosing a setup, 15 held back for the final score, and a few spares to replace rejected questions.

Examples:
  python -m src.build_golden --anchor titans
  python -m src.build_golden --anchor titans --remove q07 q12
"""

from __future__ import annotations

import argparse
import json
import random
import re

import numpy as np

from src.build_index import LoadedIndex, load_index, tokenize
from src.config import GOLDEN, PROMPTS, load_anchors, load_config
from src.generation import read_prompt
from src.judge import parse_json_list
from src.llm import chat, estimate_cost

MODEL = "gpt-4o-mini"
PROVIDER = "openai"
SEED = 42
SINGLE_TARGET = {"dev": 12, "test": 12, "spare": 5}
CROSS_TARGET = {"dev": 3, "test": 3, "spare": 1}
WINDOW = 4
MIN_LETTER_RATIO = 0.65
MAX_SIMILARITY = 0.7
MAX_ATTEMPTS_FACTOR = 4
VAGUE_REFERENCE = re.compile(
    r"\b(excerpts?|passages?|the text|the authors?|this paper|this study|this work"
    r"|the paper|the study|paper a|paper b|the method|the proposed|the described"
    r"|the first work|the second work|the two works|the model's|the module's)\b",
    re.IGNORECASE,
)


def is_readable(text: str) -> bool:
    """Skip passages that are mostly maths, tables or reference fragments."""
    letters = sum(character.isalpha() for character in text)
    return bool(text) and letters / len(text) >= MIN_LETTER_RATIO


def is_usable(item: dict) -> bool:
    """A question must stand alone, because retrieval searches every paper in the corpus."""
    question = str(item.get("question", ""))
    answer = str(item.get("answer", ""))
    if len(question) <= 20 or len(answer) <= 50 or not question.rstrip().endswith("?"):
        return False
    return not VAGUE_REFERENCE.search(question)


def too_similar(question: str, existing: list[str]) -> bool:
    words = set(tokenize(question))
    for other in existing:
        others = set(tokenize(other))
        if words and len(words & others) / len(words | others) > MAX_SIMILARITY:
            return True
    return False


def paper_windows(index: LoadedIndex) -> dict[str, list[list[dict]]]:
    """Group each paper's passages into readable runs of neighbouring passages."""
    by_paper: dict[str, list[dict]] = {}
    for chunk in index.chunks:
        by_paper.setdefault(chunk["paper_id"], []).append(chunk)
    windows: dict[str, list[list[dict]]] = {}
    for paper_id, chunks in by_paper.items():
        windows[paper_id] = []
        for start in range(0, max(len(chunks) - WINDOW + 1, 1), WINDOW):
            window = chunks[start : start + WINDOW]
            if is_readable(" ".join(chunk["text"] for chunk in window)):
                windows[paper_id].append(window)
    return windows


def related_window(window: list[dict], index: LoadedIndex) -> list[dict] | None:
    """Find the closest passages in a different paper, so a comparison can be meaningful."""
    scores = index.bm25.get_scores(tokenize(" ".join(c["text"] for c in window))[:80])
    for position in np.argsort(scores)[::-1][:50]:
        chunk = index.chunks[int(position)]
        if chunk["paper_id"] != window[0]["paper_id"] and is_readable(chunk["text"]):
            neighbours = [c for c in index.chunks if c["paper_id"] == chunk["paper_id"]]
            start = neighbours.index(chunk)
            return neighbours[max(start - 1, 0) : start + 2]
    return None


def format_excerpt(window: list[dict]) -> str:
    return f"Paper: {window[0]['title']}\n" + " ".join(chunk["text"] for chunk in window)


def ask(prompt_name: str, replacements: dict[str, str]) -> tuple[list[dict], float]:
    _, template = read_prompt(PROMPTS / "golden" / f"{prompt_name}.txt")
    for key, value in replacements.items():
        template = template.replace("{" + key + "}", value)
    result = chat(
        PROVIDER, MODEL, [{"role": "user", "content": template}], temperature=0.2, max_tokens=400
    )
    cost = estimate_cost(MODEL, result.input_tokens, result.output_tokens)
    try:
        return parse_json_list(result.text), cost
    except (ValueError, json.JSONDecodeError):
        return [], cost


def collect(
    index: LoadedIndex, kind: str, target: int, questions: list[dict], rng: random.Random
) -> float:
    """Add up to target questions of one kind, visiting the papers in turn."""
    windows = paper_windows(index)
    for paper_windows_list in windows.values():
        rng.shuffle(paper_windows_list)
    papers = [paper for paper, items in windows.items() if items]
    rng.shuffle(papers)

    cost, added, attempts = 0.0, 0, 0
    while added < target and attempts < target * MAX_ATTEMPTS_FACTOR:
        paper = papers[attempts % len(papers)]
        attempts += 1
        if not windows[paper]:
            continue
        window = windows[paper].pop()
        if kind == "single":
            sources = [window]
            items, call_cost = ask("single", {"excerpt": format_excerpt(window)})
        else:
            other = related_window(window, index)
            if other is None:
                continue
            sources = [window, other]
            items, call_cost = ask(
                "cross",
                {"excerpt_a": format_excerpt(window), "excerpt_b": format_excerpt(other)},
            )
        cost += call_cost
        existing = [question["question"] for question in questions]
        if not items or not is_usable(items[0]) or too_similar(items[0]["question"], existing):
            continue
        questions.append(
            {
                "question": items[0]["question"].strip(),
                "reference_answer": items[0]["answer"].strip(),
                "type": kind,
                "source_papers": [source[0]["paper_id"] for source in sources],
                "source_titles": [source[0]["title"] for source in sources],
            }
        )
        added += 1
    return cost


def assign_splits(questions: list[dict], rng: random.Random) -> list[dict]:
    """Split each question type separately so dev and test both contain comparisons."""
    for kind, targets in (("single", SINGLE_TARGET), ("cross", CROSS_TARGET)):
        group = [question for question in questions if question["type"] == kind]
        rng.shuffle(group)
        names = [name for name, count in targets.items() for _ in range(count)]
        for question, split in zip(group, names, strict=False):
            question["split"] = split
    kept = [question for question in questions if "split" in question]
    for number, question in enumerate(kept, start=1):
        question["id"] = f"q{number:02d}"
    return kept


def replace_questions(questions: list[dict], remove: list[str]) -> list[dict]:
    """Remove questions and promote a spare of the same type into each freed place."""
    unknown = set(remove) - {question["id"] for question in questions}
    if unknown:
        raise ValueError(f"Unknown question ids: {sorted(unknown)}")
    kept = []
    for question in questions:
        if question["id"] not in remove:
            kept.append(question)
    for question in questions:
        if question["id"] in remove and question["split"] != "spare":
            spare = next(
                (q for q in kept if q["split"] == "spare" and q["type"] == question["type"]), None
            )
            if spare is None:
                print(f"No spare left to replace {question['id']}.")
            else:
                spare["split"] = question["split"]
    return kept


def golden_path(anchor_id: str):
    return GOLDEN / f"{anchor_id}.json"


def load_golden(anchor_id: str, split: str | None = None) -> list[dict]:
    questions = json.loads(golden_path(anchor_id).read_text(encoding="utf-8"))
    return [q for q in questions if split is None or q["split"] == split]


def save_golden(anchor_id: str, questions: list[dict]) -> None:
    GOLDEN.mkdir(parents=True, exist_ok=True)
    golden_path(anchor_id).write_text(json.dumps(questions, indent=2), encoding="utf-8")


def write_review_file() -> None:
    """Write every question to one readable file for human review."""
    lines = ["# Golden questions for review", ""]
    for anchor_id in load_anchors():
        if not golden_path(anchor_id).exists():
            continue
        lines += [f"## {anchor_id}", ""]
        lines += ["| id | split | type | question | from |", "|---|---|---|---|---|"]
        for item in load_golden(anchor_id):
            source = "; ".join(title[:45] for title in item["source_titles"])
            text = item["question"].replace("|", "\\|")
            lines.append(f"| {item['id']} | {item['split']} | {item['type']} | {text} | {source} |")
        lines.append("")
    (GOLDEN / "REVIEW.md").write_text("\n".join(lines), encoding="utf-8")


def build(anchor_id: str, config: dict, force: bool = False) -> None:
    if golden_path(anchor_id).exists() and not force:
        raise FileExistsError(f"{golden_path(anchor_id)} exists. Use --force to replace it.")
    rng = random.Random(SEED)
    index = load_index(anchor_id, config)
    questions: list[dict] = []
    cost = collect(index, "single", sum(SINGLE_TARGET.values()), questions, rng)
    cost += collect(index, "cross", sum(CROSS_TARGET.values()), questions, rng)
    questions = assign_splits(questions, rng)
    save_golden(anchor_id, questions)
    counts = {name: sum(q["split"] == name for q in questions) for name in ("dev", "test", "spare")}
    print(f"{anchor_id}: {counts}, estimated cost ${cost:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    anchors = sorted(load_anchors())
    parser.add_argument("--anchor", choices=anchors, help="Default is all anchors.")
    parser.add_argument("--force", action="store_true", help="Replace an existing question set.")
    parser.add_argument("--remove", nargs="+", metavar="ID", help="Question ids to reject.")
    args = parser.parse_args()
    if args.remove:
        if not args.anchor:
            parser.error("--remove needs --anchor")
        save_golden(args.anchor, replace_questions(load_golden(args.anchor), args.remove))
    else:
        config = load_config("baseline")
        for anchor_id in [args.anchor] if args.anchor else anchors:
            build(anchor_id, config, args.force)
    write_review_file()


if __name__ == "__main__":
    main()
