import csv
import json

import pytest

from src import judge_agreement
from src.judge_agreement import agreement, choose_claims, cohen_kappa, read_labels, write_sample


def answer(question_id, verdicts, anchor="titans"):
    return {
        "anchor": anchor,
        "id": question_id,
        "question": f"Question {question_id}?",
        "context": f"[Source 1: Paper (2026)]\nPassage for {question_id}.",
        "claims": [
            {"claim": f"Claim {i} of {question_id}.", "verdict": verdict}
            for i, verdict in enumerate(verdicts)
        ],
    }


def test_cohen_kappa_matches_known_values():
    assert cohen_kappa(["SUPPORTED", "PARTIAL"], ["SUPPORTED", "PARTIAL"]) == 1.0
    # Raters who agree only as often as chance predicts score zero.
    first = ["SUPPORTED", "SUPPORTED", "NOT_SUPPORTED", "NOT_SUPPORTED"]
    second = ["SUPPORTED", "NOT_SUPPORTED", "SUPPORTED", "NOT_SUPPORTED"]
    assert cohen_kappa(first, second) == 0.0


def test_agreement_counts_partial_as_not_supported_in_the_binary_check():
    metrics = agreement(["SUPPORTED", "PARTIAL"], ["SUPPORTED", "NOT_SUPPORTED"])

    assert metrics["exact_agreement"] == 0.5
    assert metrics["supported_agreement"] == 1.0


def test_answers_the_judge_doubted_are_chosen_first(monkeypatch):
    monkeypatch.setattr(judge_agreement, "CLAIMS_PER_ANSWER", 2)
    answers = [answer(f"q{i}", ["SUPPORTED"] * 4) for i in range(10)]
    answers.append(answer("q99", ["SUPPORTED", "SUPPORTED", "NOT_SUPPORTED"]))

    chosen = choose_claims(answers, count=6)

    assert len(chosen) == 6
    # Three answers, two claims each, so labellers read only three sets of passages.
    assert len({claim["answer"]["id"] for claim in chosen}) == 3
    assert any(claim["verdict"] == "NOT_SUPPORTED" for claim in chosen)
    assert chosen == choose_claims(answers, count=6)


def test_labellers_do_not_see_the_judge_verdicts(tmp_path):
    write_sample(choose_claims([answer("q1", ["SUPPORTED", "PARTIAL"])]), tmp_path)

    sheet = (tmp_path / "to_label.csv").read_text("utf-8")
    assert "PARTIAL" not in sheet and "Claim 1 of q1." in sheet
    assert "Passage for q1." in (tmp_path / "passages.md").read_text("utf-8")
    assert json.loads((tmp_path / "judge_verdicts.json").read_text("utf-8")) == {
        "c01": "SUPPORTED",
        "c02": "PARTIAL",
    }


def label(folder, verdicts):
    path = folder / "to_label.csv"
    with open(path, encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row, verdict in zip(rows, verdicts, strict=True):
        row["human_verdict"] = verdict
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_labels_are_read_against_the_judge(tmp_path):
    write_sample(choose_claims([answer("q1", ["SUPPORTED", "PARTIAL"])]), tmp_path)
    label(tmp_path, ["supported", "NOT_SUPPORTED"])

    human, judge, _ = read_labels(tmp_path)

    assert human == ["SUPPORTED", "NOT_SUPPORTED"]
    assert judge == ["SUPPORTED", "PARTIAL"]


def test_unlabelled_claims_are_reported(tmp_path):
    write_sample(choose_claims([answer("q1", ["SUPPORTED", "PARTIAL"])]), tmp_path)
    label(tmp_path, ["SUPPORTED", "maybe"])

    with pytest.raises(ValueError, match="c02"):
        read_labels(tmp_path)


def test_the_served_setups_answers_are_loaded(tmp_path, monkeypatch):
    runs = tmp_path / "runs" / "titans"
    runs.mkdir(parents=True)
    (runs / "a.answers.jsonl").write_text(
        json.dumps({"id": "q1", "question": "Q?", "context": "C", "answer": "A"}), "utf-8"
    )
    (runs / "a.scores.jsonl").write_text(
        json.dumps({"id": "q1", "claims": [{"claim": "A claim here.", "verdict": "SUPPORTED"}]}),
        "utf-8",
    )
    served = {"variant": judge_agreement.SETUP, "split": "test", "anchor": "titans"}
    paths = {
        "answers_path": "runs/titans/a.answers.jsonl",
        "scores_path": "runs/titans/a.scores.jsonl",
    }
    index = [
        served | paths,
        served | {"variant": "B-control-baseline", "answers_path": "x", "scores_path": "x"},
    ]
    (tmp_path / "runs_index.json").write_text(json.dumps(index), "utf-8")

    answers = judge_agreement.load_judged_answers(tmp_path)

    assert [(item["anchor"], item["id"], item["context"]) for item in answers] == [
        ("titans", "q1", "C")
    ]
