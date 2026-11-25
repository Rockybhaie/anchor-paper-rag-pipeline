"""Check the LLM judge against human labels. Part of the LLM-as-a-Judge project focus.

Every score in this project comes from an LLM judge, so this measures how often people agree with it
on the served setup's held-back answers.

1. python -m src.judge_agreement sample
   Writes results/judge_agreement/to_label.csv and passages.md. The judge's verdicts are kept in a
   separate file, so labellers do not see them.
2. Read each claim's passages in passages.md and fill the human_verdict column with SUPPORTED,
   PARTIAL or NOT_SUPPORTED.
3. python -m src.judge_agreement score
   Computes agreement and Cohen's kappa and logs them to the MLflow experiment
   focus-judge-agreement.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter
from pathlib import Path

import mlflow

from src.config import RESULTS, ROOT
from src.evaluate import TRACKING_URI, get_experiment, make_artifact_paths_relative
from src.judge import VERDICT_SCORES

SETUP = "D-hybrid-free-generator"
JUDGE = {"provider": "openai", "model": "gpt-4o-mini", "prompt": "rubric"}
EXPERIMENT = "focus-judge-agreement"
FOLDER = RESULTS / "judge_agreement"
LABELS = list(VERDICT_SCORES)
CLAIMS_TO_LABEL = 40
CLAIMS_PER_ANSWER = 4
SEED = 42


def read_rows(path: Path) -> dict[str, dict]:
    lines = path.read_text(encoding="utf-8").split("\n")
    return {row["id"]: row for row in (json.loads(line) for line in lines if line.strip())}


def load_judged_answers(results: Path = RESULTS) -> list[dict]:
    """The served setup's held-back answers, each with its passages and the judge's verdicts."""
    runs = json.loads((results / "runs_index.json").read_text(encoding="utf-8"))
    answers = []
    for run in runs:
        if run.get("variant") != SETUP or run.get("split") != "test":
            continue
        texts = read_rows(results / run["answers_path"])
        scores = read_rows(results / run["scores_path"])
        for question_id, score in sorted(scores.items()):
            if score.get("claims") and question_id in texts:
                answers.append({"anchor": run["anchor"], **texts[question_id], **score})
    return answers


def choose_claims(answers: list[dict], count: int = CLAIMS_TO_LABEL) -> list[dict]:
    """Pick whole answers to label, a few claims each, so labellers read fewer passages.

    The judge supported almost every claim, so a purely random sample would hold almost no
    disagreements to check. Answers with a claim the judge doubted come first, which makes the
    sample stratified, and it is reported that way.
    """
    rng = random.Random(SEED)
    doubted = [a for a in answers if any(c["verdict"] != "SUPPORTED" for c in a["claims"])]
    others = [a for a in answers if a not in doubted]
    rng.shuffle(doubted)
    rng.shuffle(others)
    chosen = []
    for answer in doubted + others:
        if len(chosen) >= count:
            break
        claims = answer["claims"]
        # Keep the doubted claims of an answer, then fill up with its other claims.
        order = sorted(range(len(claims)), key=lambda i: claims[i]["verdict"] == "SUPPORTED")
        for i in sorted(order[: min(CLAIMS_PER_ANSWER, count - len(chosen))]):
            chosen.append({"answer": answer, "position": i + 1, **claims[i]})
    # Labelled in reading order, so each answer's claims sit together as they appear.
    return sorted(
        chosen,
        key=lambda claim: (claim["answer"]["anchor"], claim["answer"]["id"], claim["position"]),
    )


def write_sample(claims: list[dict], folder: Path = FOLDER) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    rows, verdicts, sections, contexts = [], {}, {}, {}
    for number, claim in enumerate(claims, start=1):
        answer = claim["answer"]
        key = f"{answer['anchor']}/{answer['id']}"
        item = f"c{number:02d}"
        rows.append(
            {"item": item, "answer": key, "claim": claim["claim"], "human_verdict": "", "notes": ""}
        )
        verdicts[item] = claim["verdict"]
        contexts[key] = answer["context"]
        section = sections.setdefault(
            key, [f"## {key}", f"**Question:** {answer['question']}", "", "**Claims:**"]
        )
        section.append(f"- {item}: {claim['claim']}")
    for key, section in sections.items():
        section += ["", "**Passages the judge saw:**", "", contexts[key], ""]
    with open(folder / "to_label.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (folder / "judge_verdicts.json").write_text(json.dumps(verdicts, indent=2), "utf-8")
    guide = [
        "# Claims to label",
        "",
        "For each claim, decide from the passages alone whether it is SUPPORTED, PARTIAL or "
        "NOT_SUPPORTED, and write that in the human_verdict column of to_label.csv.",
        "",
    ]
    text = "\n".join(guide + [line for section in sections.values() for line in section])
    (folder / "passages.md").write_text(text, "utf-8")


def cohen_kappa(first: list[str], second: list[str]) -> float:
    """Agreement beyond what two raters would reach by chance. 1 is perfect, 0 is chance."""
    total = len(first)
    observed = sum(a == b for a, b in zip(first, second, strict=True)) / total
    counts_a, counts_b = Counter(first), Counter(second)
    expected = sum(counts_a[label] * counts_b[label] for label in LABELS) / total**2
    return 1.0 if expected == 1 else round((observed - expected) / (1 - expected), 4)


def agreement(human: list[str], judge: list[str]) -> dict:
    def supported(labels: list[str]) -> list[str]:
        return ["yes" if label == "SUPPORTED" else "no" for label in labels]

    exact = sum(a == b for a, b in zip(human, judge, strict=True)) / len(human)
    binary = sum(a == b for a, b in zip(supported(human), supported(judge), strict=True))
    return {
        "claims_labelled": len(human),
        "exact_agreement": round(exact, 4),
        "supported_agreement": round(binary / len(human), 4),
        "cohen_kappa": cohen_kappa(human, judge),
        "judge_supported_share": round(judge.count("SUPPORTED") / len(judge), 4),
        "human_supported_share": round(human.count("SUPPORTED") / len(human), 4),
    }


def read_labels(folder: Path = FOLDER) -> tuple[list[str], list[str], list[dict]]:
    verdicts = json.loads((folder / "judge_verdicts.json").read_text("utf-8"))
    with open(folder / "to_label.csv", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    missing = [row["item"] for row in rows if row["human_verdict"].strip().upper() not in LABELS]
    if missing:
        raise ValueError(f"Label every claim with one of {LABELS}. Missing or unclear: {missing}")
    human = [row["human_verdict"].strip().upper() for row in rows]
    return human, [verdicts[row["item"]] for row in rows], rows


def score(folder: Path = FOLDER) -> dict:
    human, judge, rows = read_labels(folder)
    metrics = agreement(human, judge)
    confusion = Counter(f"human {a} / judge {b}" for a, b in zip(human, judge, strict=True))
    (folder / "confusion.json").write_text(json.dumps(dict(confusion), indent=2), "utf-8")
    os.chdir(ROOT)
    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run(experiment_id=get_experiment(EXPERIMENT), run_name="judge-vs-human"):
        mlflow.log_params(
            {f"judge.{key}": value for key, value in JUDGE.items()}
            | {"setup": SETUP, "split": "test", "sample": "stratified", "seed": SEED}
        )
        mlflow.log_metrics(metrics)
        for name in ("to_label.csv", "judge_verdicts.json", "confusion.json"):
            mlflow.log_artifact(str(folder / name))
    make_artifact_paths_relative(ROOT / "mlflow.db")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["sample", "score"])
    args = parser.parse_args()
    if args.step == "sample":
        claims = choose_claims(load_judged_answers())
        write_sample(claims)
        doubted = sum(claim["verdict"] != "SUPPORTED" for claim in claims)
        print(f"Wrote {len(claims)} claims, {doubted} of them doubted by the judge, to {FOLDER}")
    else:
        print(json.dumps(score(), indent=2))


if __name__ == "__main__":
    main()
