import json
import sqlite3

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from src import evaluate, generation, judge, retrieval
from src.evaluate import (
    Budget,
    BudgetExceeded,
    Evaluation,
    make_artifact_paths_relative,
    pick_judge,
    pick_winner,
    source_hit,
    summarize,
)
from src.llm import ChatResult
from tests.conftest import fake_embed


def run(name, faithfulness=0.9, relevancy=0.7, total_ms=1000, answered=1.0, refusal=0.0, **extra):
    metrics = {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "total_ms": total_ms,
        "answered_share": answered,
        "refusal_rate": refusal,
        "cost_usd": 0.01,
    }
    return {"name": name, "metrics": metrics | extra}


def test_winner_has_the_highest_faithfulness():
    runs = [run("a", 0.80), run("b", 0.95), run("c", 0.90)]
    assert pick_winner(runs)["name"] == "b"


def test_near_ties_go_to_higher_relevancy_then_lower_latency():
    assert pick_winner([run("a", 0.950, 0.6), run("b", 0.945, 0.8)])["name"] == "b"
    fast = run("fast", 0.95, 0.7, total_ms=500)
    assert pick_winner([run("slow", 0.95, 0.7, total_ms=900), fast])["name"] == "fast"


def test_a_clear_faithfulness_lead_beats_relevancy():
    assert pick_winner([run("a", 0.99, 0.5), run("b", 0.90, 0.9)])["name"] == "a"


def test_setups_that_rarely_answer_cannot_win():
    runs = [run("refuses", 1.0, refusal=0.5), run("silent", 1.0, answered=0.5), run("real", 0.8)]
    assert pick_winner(runs)["name"] == "real"


def test_pick_winner_explains_when_nothing_qualifies():
    with pytest.raises(ValueError, match="enough questions"):
        pick_winner([run("none", None)])


def test_judge_with_the_largest_gap_wins_not_the_highest_score():
    lenient = run("lenient", 1.0, judge_gap=0.1)
    careful = run("careful", 0.8, judge_gap=0.6)
    assert pick_judge([lenient, careful])["name"] == "careful"


def test_judge_ties_go_to_the_cheaper_one():
    costly = run("costly", judge_gap=0.6)
    costly["metrics"]["cost_usd"] = 0.5
    assert pick_judge([costly, run("cheap", judge_gap=0.595)])["name"] == "cheap"


def test_judge_that_often_fails_cannot_win():
    broken = run("broken", answered=0.5, judge_gap=0.9)
    assert pick_judge([broken, run("working", judge_gap=0.3)])["name"] == "working"


def test_most_changed_question_ignores_questions_some_runs_never_answered():
    answers = [{"q1": {}, "q2": {}, "extra": {}}, {"q1": {}, "q2": {}}]
    scores = [
        {"q1": {"faithfulness": 1.0}, "q2": {"faithfulness": 1.0}, "extra": {"faithfulness": 0.0}},
        {"q1": {"faithfulness": 0.5}, "q2": {"faithfulness": 1.0}},
    ]
    assert evaluate.most_changed_question(answers, scores) == "q1"


def test_most_changed_question_counts_a_refusal_as_a_change():
    answers = [{"q1": {}, "q2": {}}, {"q1": {}, "q2": {}}]
    scores = [
        {"q1": {"faithfulness": 0.9}, "q2": {"faithfulness": 0.9}},
        {"q1": {"faithfulness": 1.0}, "q2": {"faithfulness": None}},
    ]
    assert evaluate.most_changed_question(answers, scores) == "q2"


def test_most_changed_question_ignores_questions_every_run_refused():
    answers = [{"q1": {}, "q2": {}}, {"q1": {}, "q2": {}}]
    scores = [
        {"q1": {"faithfulness": None}, "q2": {"faithfulness": 0.9}},
        {"q1": {"faithfulness": None}, "q2": {"faithfulness": 1.0}},
    ]
    assert evaluate.most_changed_question(answers, scores) == "q2"


def test_saved_rows_survive_unusual_line_breaks(tmp_path):
    path = tmp_path / "rows.jsonl"
    output = evaluate.RowFile(path)
    output.append({"id": "q1", "answer": "first second\u0085third"})
    output.append({"id": "q2", "answer": "plain"})
    rows = evaluate.load_rows(path)
    assert rows["q1"]["answer"] == "first second\u0085third"
    assert set(rows) == {"q1", "q2"}


def test_source_hit_counts_each_source_paper():
    row = {"source_papers": ["a", "b"], "retrieved_papers": ["a", "a", "c"]}
    assert source_hit(row) == 0.5


def answer(row_id, text="An answer.", papers=("p1",)):
    return {
        "id": row_id,
        "answer": text,
        "source_papers": ["p1"],
        "retrieved_papers": list(papers),
        "retrieval_ms": 100.0,
        "generation_ms": 900.0,
        "cost": 0.001,
    }


def test_summarize_leaves_failures_and_claimless_answers_out_of_faithfulness():
    answers = [answer("q1"), answer("q2"), answer("q3", "Not enough information to say.")]
    scores = [
        {"id": "q1", "faithfulness": 0.5, "relevancy": 0.8, "cost": 0.01},
        {"id": "q2", "error": "ValueError: bad reply", "cost": 0.01},
        {"id": "q3", "faithfulness": None, "relevancy": 0.2, "cost": 0.01},
    ]

    metrics = summarize(answers, scores)

    assert metrics["faithfulness"] == 0.5
    assert metrics["relevancy"] == pytest.approx(0.5)
    assert metrics["failures"] == 1
    assert metrics["answered_share"] == pytest.approx(1 / 3)
    assert metrics["refusal_rate"] == pytest.approx(1 / 3)
    assert metrics["total_ms"] == 1000
    assert metrics["cost_usd"] == pytest.approx(0.033)


def test_summarize_counts_a_reply_with_no_claims_as_a_refusal():
    # This wording has no known refusal phrase, but the judge finds no factual claims in it.
    answers = [answer("q1", "The excerpts do not mention FPRM."), answer("q2")]
    scores = [
        {"id": "q1", "faithfulness": None, "relevancy": 0.9, "cost": 0},
        {"id": "q2", "faithfulness": 1.0, "relevancy": 0.6, "cost": 0},
    ]

    metrics = summarize(answers, scores)

    assert metrics["refusal_rate"] == 0.5
    assert metrics["relevancy"] == pytest.approx(0.75)
    assert metrics["relevancy_when_answered"] == 0.6


def test_summarize_reports_the_judge_gap():
    scores = [{"id": "q1", "faithfulness": 0.9, "relevancy": 0.5, "cost": 0,
               "wrong_context_faithfulness": 0.2}]
    assert summarize([answer("q1")], scores)["judge_gap"] == pytest.approx(0.7)


def test_budget_stops_at_the_limit_and_remembers_spend(tmp_path):
    path = tmp_path / "spend.json"
    budget = Budget(path, limit=1.0)
    budget.add(0.6)
    with pytest.raises(BudgetExceeded):
        budget.add(0.6)
    assert Budget(path, limit=1.0).spent == pytest.approx(1.2)


def test_make_artifact_paths_relative(tmp_path):
    database = tmp_path / "mlflow.db"
    connection = sqlite3.connect(database)
    connection.execute("create table experiments (artifact_location text)")
    connection.execute("create table runs (artifact_uri text)")
    connection.execute("insert into experiments values ('file:///D:/x y/mlartifacts/stage-1')")
    connection.execute("insert into runs values ('file:///D:/x y/mlartifacts/s1/abc/artifacts')")
    connection.commit()
    connection.close()

    make_artifact_paths_relative(database)

    connection = sqlite3.connect(database)
    assert connection.execute("select * from experiments").fetchone()[0] == "./mlartifacts/stage-1"
    assert connection.execute("select * from runs").fetchone()[0] == (
        "./mlartifacts/s1/abc/artifacts"
    )
    connection.close()


QUESTIONS = [
    {
        "id": f"q{i}",
        "question": f"How is memory used for robot control number {i}?",
        "type": "single",
        "source_papers": ["01_citing"],
        "split": split,
    }
    for i, split in enumerate(["dev", "dev", "test", "test"])
]

STAGE = {
    "id": 1,
    "experiment": "stage-test",
    "anchor": "titans",
    "groups": [
        {
            "setting": "judge",
            "kind": "judge",
            "variants": [
                {"name": "lenient", "overrides": {"judge": {"model": "lenient"}}},
                {"name": "careful", "overrides": {"judge": {"model": "careful"}}},
            ],
        },
        {
            "setting": "prompt",
            "variants": [
                {"name": "p0", "overrides": {"generation": {"prompt": "p0"}}},
                {"name": "p1", "overrides": {"generation": {"prompt": "p1"}}},
            ],
        },
    ],
}


@pytest.fixture
def harness(corpus, store, config, monkeypatch, tmp_path):
    """Run the real evaluation code with stand-ins for every outside service."""
    from src.build_index import build

    build("titans", config, store)
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    monkeypatch.setattr(retrieval, "embed_texts", fake_embed)
    monkeypatch.setattr(retrieval, "rerank_scores", lambda q, texts: [0.5] * len(texts))
    monkeypatch.setattr(evaluate, "load_golden", lambda anchor, split: [
        q for q in QUESTIONS if q["split"] == split
    ])
    monkeypatch.setattr(evaluate, "load_anchors", lambda: {"titans": {}})
    monkeypatch.setattr(judge, "embed_texts", lambda texts, cfg: fake_embed(texts))
    calls = {"generation": 0, "judge": 0}

    def fake_generation_chat(provider, model, messages, temperature, max_tokens):
        calls["generation"] += 1
        return ChatResult("Memory guides control (Source 1).", 100, 20, 0.1)

    def fake_judge_chat(provider, model, messages, temperature, max_tokens):
        calls["judge"] += 1
        content = messages[0]["content"]
        if content.startswith("Extract all"):
            return ChatResult('["Memory guides robot control tasks."]', 100, 20, 0.1)
        if content.startswith("Generate exactly"):
            return ChatResult('["How is memory used for control?"]', 100, 20, 0.1)
        # The careful judge rejects claims against passages that do not fit, the other never does.
        supported = model == "lenient" or "robot" in content.split("Claims:")[0][:800]
        verdict = "SUPPORTED" if supported else "NOT_SUPPORTED"
        return ChatResult(json.dumps([{"id": 1, "verdict": verdict}]), 100, 20, 0.1)

    monkeypatch.setattr(generation, "chat", fake_generation_chat)
    monkeypatch.setattr(judge, "chat", fake_judge_chat)
    evaluation = Evaluation(store, results=tmp_path / "results")
    evaluation.baseline = config
    return {"evaluation": evaluation, "calls": calls, "store": store, "config": config}


def test_stage_logs_runs_with_params_metrics_and_artifacts(harness):
    evaluation = harness["evaluation"]

    evaluation.run_stage(STAGE)

    client = MlflowClient()
    experiment = client.get_experiment_by_name("stage-test")
    runs = client.search_runs([experiment.experiment_id])
    assert {r.info.run_name for r in runs} == {
        "s1-judge-lenient",
        "s1-judge-careful",
        "s1-prompt-p0",
        "s1-prompt-p1",
    }
    prompt_run = next(r for r in runs if r.info.run_name == "s1-prompt-p1")
    assert prompt_run.data.params["generation.prompt"] == "p1"
    assert prompt_run.data.params["setting_changed"] == "prompt"
    assert "faithfulness" in prompt_run.data.metrics
    assert client.list_artifacts(prompt_run.info.run_id)
    assert sum(r.data.tags.get("winner") == "true" for r in runs) == 2


def test_judge_winner_scores_every_later_run(harness):
    evaluation = harness["evaluation"]
    evaluation.run_stage(STAGE)
    assert evaluation.winners["overrides"]["judge"]["model"] in {"lenient", "careful"}
    later = [r for r in evaluation.index if r["setting"] == "prompt"]
    assert len(later) == 2
    assert all("judge_gap" not in r["metrics"] for r in later)


def test_restarting_a_stage_spends_nothing_and_logs_no_duplicates(harness):
    evaluation = harness["evaluation"]
    evaluation.run_stage(STAGE)
    calls_after_first = dict(harness["calls"])
    spent = evaluation.budget.spent

    evaluation.reset_winners()
    evaluation.run_stage(STAGE)

    assert harness["calls"] == calls_after_first
    assert evaluation.budget.spent == spent
    client = MlflowClient()
    experiment = client.get_experiment_by_name("stage-test")
    assert len(client.search_runs([experiment.experiment_id])) == 4


def test_a_stopped_run_continues_from_the_saved_questions(harness):
    evaluation = harness["evaluation"]
    calls = harness["calls"]
    real = generation.chat

    def stop_on_second(*args, **kwargs):
        if calls["generation"] == 1:
            raise ConnectionError("network down")
        return real(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(generation, "chat", stop_on_second)
        with pytest.raises(ConnectionError):
            evaluation.run_stage(STAGE)
    assert calls["generation"] == 1

    evaluation.reset_winners()
    evaluation.run_stage(STAGE)

    # One saved answer plus the rest: two questions per prompt, with p0 shared by the judge test.
    assert calls["generation"] == 4


def test_saved_judge_failures_are_tried_again(harness, tmp_path):
    answers = [{"id": "q1", "question": "A question?", "answer": "An answer.", "context": "text"}]
    path = tmp_path / "scores.jsonl"
    evaluate.RowFile(path).append({"id": "q1", "error": "JSONDecodeError: bad reply", "cost": 0.0})
    settings = {"provider": "openai", "model": "lenient", "prompt": "binary"}

    rows = evaluate.score_answers(answers, settings, harness["evaluation"].budget, path)

    assert "error" not in rows[0]
    assert rows[0]["faithfulness"] == 1.0
    assert "error" not in evaluate.load_rows(path)["q1"]


def test_run_output_holds_only_that_runs_questions(tmp_path):
    answers = [
        {
            "id": "q1",
            "question": "Q?",
            "type": "single",
            "source_papers": ["a"],
            "retrieved_papers": ["a"],
            "answer": "A.",
            "context": "long passages left out of this file",
        }
    ]
    scores = [{"id": "q1", "faithfulness": 1.0, "relevancy": 0.5, "cost": 0.0}]

    path = evaluate.write_run_output(tmp_path / "out" / "run.jsonl", answers, scores)

    rows = evaluate.load_rows(path)
    assert set(rows) == {"q1"}
    assert rows["q1"]["faithfulness"] == 1.0
    assert "context" not in rows["q1"]


def test_final_compares_setups_on_held_back_questions(harness, monkeypatch, tmp_path):
    evaluation = harness["evaluation"]
    monkeypatch.setattr(evaluate, "CONFIGS", tmp_path)
    evaluation.run_stage(STAGE)
    evaluation.winners["best_free_generation"] = {"provider": "groq", "model": "free-model"}

    best = evaluation.run_final({"experiment": "final-test"})

    client = MlflowClient()
    experiment = client.get_experiment_by_name("final-test")
    runs = client.search_runs([experiment.experiment_id])
    parents = [r for r in runs if "mlflow.parentRunId" not in r.data.tags]
    children = [r for r in runs if "mlflow.parentRunId" in r.data.tags]
    assert len(parents) == len(evaluation.final_setups())
    assert len(children) == len(parents)
    assert all(r.data.params["split"] == "test" for r in runs)
    assert best["name"] in evaluation.final_setups()
    assert (tmp_path / "results" / "best_run.json").exists()
    serving = (tmp_path / "serving.yaml").read_text(encoding="utf-8")
    assert "judge" not in serving
    assert "Served setup" in serving

    evaluation.write_reports()
    assert (tmp_path / "results" / "summary.csv").exists()
    assert "Same question" in (tmp_path / "results" / "side_by_side.md").read_text("utf-8")


def setup(name, mode, faithfulness):
    return run(name, faithfulness) | {"config": {"retrieval": {"mode": mode}}}


def test_serving_prefers_meaning_search_when_it_is_close_to_the_top_score():
    keyword = setup("keyword", "keyword", 0.98)
    hybrid = setup("hybrid", "hybrid", 0.97)
    assert evaluate.choose_serving([keyword, hybrid], keyword, 0.02)["name"] == "hybrid"


def test_serving_keeps_the_top_score_when_meaning_search_is_far_behind():
    keyword = setup("keyword", "keyword", 0.98)
    hybrid = setup("hybrid", "hybrid", 0.90)
    assert evaluate.choose_serving([keyword, hybrid], keyword, 0.02)["name"] == "keyword"


def test_serving_is_the_best_setup_when_all_use_keyword_search():
    keyword = setup("keyword", "keyword", 0.98)
    assert evaluate.choose_serving([keyword], keyword, 0.02)["name"] == "keyword"


def test_extra_setups_build_on_the_winners(harness):
    evaluation = harness["evaluation"]
    evaluation.winners["overrides"] = {"generation": {"prompt": "p1"}}
    extras = [{"name": "D-hybrid", "overrides": {"retrieval": {"mode": "hybrid"}}}]

    setups = evaluation.final_setups(extras)

    assert setups["D-hybrid"]["retrieval"]["mode"] == "hybrid"
    assert setups["D-hybrid"]["generation"]["prompt"] == "p1"


def test_final_setups_drop_duplicates_and_keep_the_baseline_control(harness):
    evaluation = harness["evaluation"]
    evaluation.winners = {"overrides": {}, "best_free_generation": None}
    assert list(evaluation.final_setups()) == ["A-all-winners"]

    evaluation.winners["overrides"] = {"generation": {"prompt": "p1"}}
    assert list(evaluation.final_setups()) == ["A-all-winners", "B-control-baseline"]
    assert evaluation.final_setups()["B-control-baseline"]["generation"]["prompt"] == "p0"
