"""Run the staged evaluation and log every run to MLflow.

For a RAG system this replaces training cycles: each run changes a prompt, model, parameter,
temperature or judge, then scores the answers. Stages use the dev questions. The final comparison
uses the held-back test questions on all four anchors.

Finished work is saved question by question, so a stopped run continues where it left off.

Examples:
  python -m src.evaluate                 Run everything.
  python -m src.evaluate --split dev --stage 2    Run stages 1 and 2.
  python -m src.evaluate --split test    Score the final setups on the held-back questions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import mlflow
import yaml
from mlflow.tracking import MlflowClient

from src.build_golden import load_golden
from src.build_index import build, load_index, setup_name
from src.config import CONFIGS, RESULTS, ROOT, apply_overrides, load_anchors, load_config, load_yaml
from src.generation import format_context, generate, is_refusal
from src.judge import Judge
from src.retrieval import retrieve
from src.vector_store import PineconeStore

TRACKING_URI = "sqlite:///mlflow.db"
ARTIFACT_FOLDER = "mlartifacts"
PRIMARY_METRIC = "faithfulness"
BUDGET_USD = 3.0
MAX_WORKERS = 4
TIE_MARGIN = 0.01
# A setup that often fails or refuses could otherwise score well by saying very little.
MIN_ANSWERED_SHARE = 0.8
MAX_REFUSAL_RATE = 0.34
MIN_JUDGE_SUCCESS_SHARE = 0.8
RUN_METRICS = (
    PRIMARY_METRIC,
    "relevancy",
    "relevancy_when_answered",
    "source_hit_rate",
    "refusal_rate",
    "failures",
    "retrieval_ms",
    "generation_ms",
    "total_ms",
    "cost_usd",
    "wrong_context_faithfulness",
    "judge_gap",
)


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Keep a running total of estimated spend, and stop before it passes the limit."""

    def __init__(self, path: Path, limit: float = BUDGET_USD):
        self.path = path
        self.limit = limit
        self.lock = threading.Lock()
        self.spent = 0.0
        if path.exists():
            self.spent = json.loads(path.read_text(encoding="utf-8"))["estimated_usd"]

    def add(self, cost: float) -> None:
        with self.lock:
            self.spent += cost
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"estimated_usd": round(self.spent, 5)}), "utf-8")
            if self.spent > self.limit:
                raise BudgetExceeded(f"Estimated spend ${self.spent:.2f} passed ${self.limit:.2f}.")


def short_hash(value: object) -> str:
    text = json.dumps(value, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def answer_key(anchor_id: str, config: dict) -> str:
    sections = ("chunking", "embedding", "retrieval", "generation")
    return f"{anchor_id}-{short_hash({key: config[key] for key in sections})}"


def load_rows(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    # Not splitlines(): answers can contain characters like U+2028 that it would also split on.
    lines = path.read_text(encoding="utf-8").split("\n")
    rows = [json.loads(line) for line in lines if line.strip()]
    return {row["id"]: row for row in rows}


class RowFile:
    """Save each finished question at once, so stopping loses almost nothing."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        with self.lock, open(self.path, "a", encoding="utf-8") as file:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def answer_questions(
    anchor_id: str, questions: list[dict], config: dict, store, budget: Budget, run_dir: Path
) -> list[dict]:
    """Retrieve and generate an answer for each question not already saved."""
    path = run_dir / f"{answer_key(anchor_id, config)}.answers.jsonl"
    done = load_rows(path)
    pending = [question for question in questions if question["id"] not in done]
    if pending:
        index = load_index(anchor_id, config)
        output = RowFile(path)
        # Retrieval runs one at a time so its measured latency is not inflated by other questions.
        retrieved = []
        for question in pending:
            result = retrieve(question["question"], index, config, store)
            budget.add(result.cost)
            retrieved.append((question, result))

        def write_answer(item) -> None:
            question, result = item
            answer = generate(question["question"], result.passages, config["generation"])
            row = {
                "id": question["id"],
                "question": question["question"],
                "type": question["type"],
                "source_papers": question["source_papers"],
                "retrieved_papers": [passage["paper_id"] for passage in result.passages],
                "context": format_context(result.passages),
                "answer": answer.text,
                "retrieval_ms": result.latency_s * 1000,
                "generation_ms": answer.latency_s * 1000,
                "input_tokens": answer.input_tokens,
                "output_tokens": answer.output_tokens,
                "cost": answer.cost + result.cost,
                "temperature_applied": answer.temperature_applied,
            }
            # Saved before the budget check, so a paid answer is never thrown away.
            output.append(row)
            done[row["id"]] = row
            budget.add(answer.cost)

        with ThreadPoolExecutor(MAX_WORKERS) as pool:
            list(pool.map(write_answer, retrieved))
    return [done[question["id"]] for question in questions]


def score_answers(
    answers: list[dict],
    judge_settings: dict,
    budget: Budget,
    path: Path,
    wrong_context: bool = False,
) -> list[dict]:
    """Score each answer not already saved. Unreadable judge replies are recorded as failures.

    Saved failures are tried again, so a fix to the judge applies without repeating other work.
    """
    done = {key: row for key, row in load_rows(path).items() if "error" not in row}
    output = RowFile(path)

    def score(position: int) -> None:
        answer = answers[position]
        if answer["id"] in done:
            return
        judge = Judge(judge_settings)
        row = {"id": answer["id"]}
        try:
            result = judge.score(answer["question"], answer["answer"], answer["context"])
            row |= {
                "faithfulness": result.faithfulness,
                "relevancy": result.relevancy,
                "claims": result.claims,
                "generated_questions": result.generated_questions,
            }
            if wrong_context and result.claims:
                # Passages for a different question should support far fewer of the claims.
                other = answers[(position + len(answers) // 2) % len(answers)]["context"]
                claims = [item["claim"] for item in result.claims]
                row["wrong_context_faithfulness"] = judge.verify_claims(claims, other)[0]
        except (ValueError, KeyError, TypeError) as error:
            row["error"] = f"{type(error).__name__}: {error}"[:300]
        row["cost"] = judge.cost
        output.append(row)
        done[row["id"]] = row
        budget.add(judge.cost)

    with ThreadPoolExecutor(MAX_WORKERS) as pool:
        list(pool.map(score, range(len(answers))))
    return [done[answer["id"]] for answer in answers]


def write_run_output(path: Path, answers: list[dict], scores: list[dict]) -> Path:
    """Save one readable file per run, holding only that run's questions, for MLflow."""
    keep = ("id", "question", "type", "source_papers", "retrieved_papers", "answer")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        for answer, score in zip(answers, scores, strict=True):
            row = {key: answer[key] for key in keep} | {
                key: value for key, value in score.items() if key != "id"
            }
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def source_hit(answer: dict) -> float:
    """The share of a question's source papers that appear among the retrieved passages."""
    found = sum(paper in answer["retrieved_papers"] for paper in answer["source_papers"])
    return found / len(answer["source_papers"])


def refused_to_answer(answer: dict, score: dict) -> bool:
    """A refusal uses a known phrase, or the judge found no factual claims in the answer."""
    made_no_claims = "error" not in score and score.get("faithfulness") is None
    return made_no_claims or is_refusal(answer["answer"])


def summarize(answers: list[dict], scores: list[dict]) -> dict:
    """Turn per-question rows into the metrics logged for one run."""
    scored = [row for row in scores if "error" not in row]
    faithful = [row["faithfulness"] for row in scored if row["faithfulness"] is not None]
    wrong = [
        row["wrong_context_faithfulness"] for row in scored if "wrong_context_faithfulness" in row
    ]
    refused = [refused_to_answer(a, s) for a, s in zip(answers, scores, strict=True)]
    answered = [s for s, was_refused in zip(scores, refused, strict=True) if not was_refused]
    metrics = {
        "faithfulness": mean(faithful),
        "relevancy": mean([row["relevancy"] for row in scored]),
        # A refusal repeats the question's words, so it can score well on plain relevancy.
        "relevancy_when_answered": mean([r["relevancy"] for r in answered if "error" not in r]),
        "source_hit_rate": mean([source_hit(answer) for answer in answers]),
        "refusal_rate": mean([float(was_refused) for was_refused in refused]),
        "failures": float(len(scores) - len(scored)),
        "answered_share": len(faithful) / len(answers) if answers else 0.0,
        "retrieval_ms": mean([answer["retrieval_ms"] for answer in answers]),
        "generation_ms": mean([answer["generation_ms"] for answer in answers]),
        "cost_usd": sum(answer["cost"] for answer in answers) + sum(r["cost"] for r in scores),
    }
    metrics["total_ms"] = (metrics["retrieval_ms"] or 0) + (metrics["generation_ms"] or 0)
    if wrong and metrics["faithfulness"] is not None:
        metrics["wrong_context_faithfulness"] = mean(wrong)
        metrics["judge_gap"] = metrics["faithfulness"] - metrics["wrong_context_faithfulness"]
    return metrics


def pick_winner(runs: list[dict]) -> dict:
    """Highest faithfulness. Within the tie margin, higher relevancy, then lower latency."""
    eligible = [
        run
        for run in runs
        if run["metrics"]["faithfulness"] is not None
        and run["metrics"]["answered_share"] >= MIN_ANSWERED_SHARE
        and run["metrics"]["refusal_rate"] <= MAX_REFUSAL_RATE
    ]
    if not eligible:
        raise ValueError("No run answered enough questions to be chosen.")
    best = max(run["metrics"]["faithfulness"] for run in eligible)
    close = [run for run in eligible if run["metrics"]["faithfulness"] >= best - TIE_MARGIN]
    return max(close, key=lambda r: (r["metrics"]["relevancy"] or 0, -r["metrics"]["total_ms"]))


def pick_judge(runs: list[dict]) -> dict:
    """The judge that best tells real passages from unrelated ones. Ties go to the cheaper one."""
    eligible = [
        run
        for run in runs
        if run["metrics"].get("judge_gap") is not None
        and run["metrics"]["answered_share"] >= MIN_JUDGE_SUCCESS_SHARE
    ]
    if not eligible:
        raise ValueError("No judge scored enough answers to be chosen.")
    best = max(run["metrics"]["judge_gap"] for run in eligible)
    close = [run for run in eligible if run["metrics"]["judge_gap"] >= best - TIE_MARGIN]
    return min(close, key=lambda run: run["metrics"]["cost_usd"])


def choose_serving(results: list[dict], best: dict, margin: float) -> dict:
    """Prefer a setup that searches by meaning if it is within margin of the top score."""
    with_meaning = [run for run in results if run["config"]["retrieval"]["mode"] != "keyword"]
    if not with_meaning:
        return best
    candidate = pick_winner(with_meaning)
    close = candidate["metrics"]["faithfulness"] >= best["metrics"]["faithfulness"] - margin
    return candidate if close else best


def most_changed_question(answers: list[dict], scores: list[dict]) -> str:
    """Among questions every run answered, find the one whose faithfulness varied the most."""
    shared = set.intersection(*(set(rows) for rows in answers), *(set(rows) for rows in scores))

    def spread(question_id: str) -> tuple[float, str]:
        values = [rows[question_id].get("faithfulness") for rows in scores]
        known = [value for value in values if value is not None]
        if not known:
            return 0.0, question_id
        # Some runs answering while others make no claims, usually refusing, is a visible change.
        return max(known) - min(known) + 0.5 * values.count(None), question_id

    return max(shared, key=spread)


def flatten(config: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in config.items()
        for key, value in values.items()
    }


def get_experiment(name: str) -> str:
    experiment = mlflow.get_experiment_by_name(name)
    if experiment:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=f"{ARTIFACT_FOLDER}/{name}")


def log_run(
    experiment: str,
    run_name: str,
    run_key: str,
    config: dict,
    params: dict,
    tags: dict,
    metrics: dict,
    artifacts: list[Path],
    nested: bool = False,
) -> str:
    """Log one run, unless it was already logged by an earlier attempt."""
    experiment_id = get_experiment(experiment)
    found = MlflowClient().search_runs(
        [experiment_id], f"tags.run_key = '{run_key}' and attributes.status = 'FINISHED'"
    )
    if found:
        return found[0].info.run_id
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=nested) as run:
        mlflow.log_params(flatten(config) | params)
        mlflow.set_tags(tags | {"run_key": run_key, "primary_metric": PRIMARY_METRIC})
        mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})
        for artifact in artifacts:
            if artifact.exists():
                mlflow.log_artifact(str(artifact))
        return run.info.run_id


def mark_winner(run_id: str, note: str) -> None:
    client = MlflowClient()
    client.set_tag(run_id, "winner", "true")
    client.set_tag(run_id, "mlflow.note.content", note)


def make_artifact_paths_relative(database: Path) -> None:
    """MLflow saves full paths. Relative ones keep artifacts working after a clone."""
    marker = f"/{ARTIFACT_FOLDER}/"
    connection = sqlite3.connect(database)
    with connection:
        for table, column in (("experiments", "artifact_location"), ("runs", "artifact_uri")):
            query = f"select rowid, {column} from {table} where {column} like ?"
            for rowid, value in connection.execute(query, (f"%{marker}%",)).fetchall():
                relative = f"./{ARTIFACT_FOLDER}/" + value.split(marker, 1)[1]
                connection.execute(
                    f"update {table} set {column} = ? where rowid = ?", (relative, rowid)
                )
    connection.close()


class Evaluation:
    """Holds what the stages share: the store, the budget, the winners so far, and past runs."""

    def __init__(self, store, results: Path = RESULTS):
        self.store = store
        self.results = results
        self.run_dir = results / "runs"
        self.outputs_dir = results / "run_outputs"
        self.budget = Budget(results / "spend.json")
        self.baseline = load_config("baseline")
        self.winners_path = results / "winners.json"
        self.winners = {"overrides": {}, "best_free_generation": None}
        if self.winners_path.exists():
            self.winners = json.loads(self.winners_path.read_text(encoding="utf-8"))
        self.index_path = results / "runs_index.json"
        self.index = []
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))

    def reset_winners(self) -> None:
        """Start the stages from the baseline. Replaying finished stages costs nothing."""
        self.winners = {"overrides": {}, "best_free_generation": None}

    def current_config(self) -> dict:
        return apply_overrides(self.baseline, self.winners["overrides"])

    def carry(self, overrides: dict) -> None:
        for section, values in overrides.items():
            self.winners["overrides"].setdefault(section, {}).update(values)
        self.winners_path.parent.mkdir(parents=True, exist_ok=True)
        self.winners_path.write_text(json.dumps(self.winners, indent=2), "utf-8")

    def ensure_index(self, anchor_id: str, config: dict) -> None:
        try:
            load_index(anchor_id, config)
        except FileNotFoundError:
            self.budget.add(build(anchor_id, config, self.store)["cost"])

    def record(self, entry: dict) -> None:
        """Remember a run for the reports, replacing any earlier record of it."""
        self.index = [item for item in self.index if item["run_key"] != entry["run_key"]]
        self.index.append(entry)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.index, indent=1), "utf-8")

    def run_variant(
        self, stage: dict, group: dict, variant: dict, base: dict, split: str, questions: list[dict]
    ) -> dict:
        anchor_id = stage["anchor"]
        config = apply_overrides(base, variant["overrides"])
        self.ensure_index(anchor_id, config)
        run_dir = self.run_dir / anchor_id
        answers = answer_questions(anchor_id, questions, config, self.store, self.budget, run_dir)

        is_judge_group = group.get("kind") == "judge"
        key = answer_key(anchor_id, config)
        score_id = short_hash(config["judge"])
        score_path = run_dir / f"{key}.scores-{score_id}.jsonl"
        scores = score_answers(answers, config["judge"], self.budget, score_path, is_judge_group)
        metrics = summarize(answers, scores)
        if not is_judge_group:
            # A later stage can reuse these scores, but this check only belongs to the judge test.
            metrics.pop("wrong_context_faithfulness", None)
            metrics.pop("judge_gap", None)

        run_name = f"s{stage['id']}-{group['setting']}-{variant['name']}"
        run_key = short_hash([run_name, key, score_id])
        params = {
            "anchor": anchor_id,
            "split": split,
            "questions": len(questions),
            "setting_changed": group["setting"],
            "setting_value": variant["name"],
            "index_setup": setup_name(config),
        }
        tags = {"stage": str(stage["id"]), "anchor": anchor_id, "split": split}
        tags |= {"setting": group["setting"]}
        output = write_run_output(self.outputs_dir / f"{run_name}.jsonl", answers, scores)
        run_id = log_run(
            stage["experiment"], run_name, run_key, config, params, tags, metrics, [output]
        )
        self.report(run_name, metrics)
        run = {
            "stage": stage["id"],
            "anchor": anchor_id,
            "setting": group["setting"],
            "variant": variant["name"],
            "run_key": run_key,
            "run_id": run_id,
            "split": split,
            "answers_path": (run_dir / f"{key}.answers.jsonl").relative_to(self.results).as_posix(),
            "scores_path": score_path.relative_to(self.results).as_posix(),
            "metrics": metrics,
            "overrides": variant["overrides"],
            "generation": config["generation"],
            "winner": False,
        }
        self.record(run)
        return run

    @staticmethod
    def report(run_name: str, metrics: dict) -> None:
        faithfulness = metrics["faithfulness"]
        shown = "n/a" if faithfulness is None else f"{faithfulness:.3f}"
        print(
            f"  {run_name}: faithfulness {shown}, relevancy {metrics['relevancy'] or 0:.3f}, "
            f"source hit {metrics['source_hit_rate']:.2f}, cost ${metrics['cost_usd']:.3f}",
            flush=True,
        )

    def run_group(self, stage: dict, group: dict, questions: list[dict]) -> dict:
        base = self.current_config()
        if group.get("pin"):
            base = apply_overrides(base, group["pin"])
        runs = [
            self.run_variant(stage, group, variant, base, "dev", questions)
            for variant in group["variants"]
        ]
        winner = pick_judge(runs) if group.get("kind") == "judge" else pick_winner(runs)
        winner["winner"] = True
        self.record(winner)
        mark_winner(winner["run_id"], f"Chosen {group['setting']} for later stages.")
        self.carry(winner["overrides"])
        if group.get("track_best_free"):
            free = [run for run in runs if run["generation"]["provider"] != "openai"]
            best_free = pick_winner(free)
            self.winners["best_free_generation"] = best_free["overrides"]["generation"]
            self.carry({})
        print(f"  chosen {group['setting']}: {winner['variant']}", flush=True)
        return winner

    def run_stage(self, stage: dict) -> None:
        print(f"Stage {stage['id']} on {stage['anchor']}", flush=True)
        questions = load_golden(stage["anchor"], "dev")
        for group in stage["groups"]:
            self.run_group(stage, group, questions)

    def final_setups(self, extra_setups: list[dict] | None = None) -> dict[str, dict]:
        """A: all winners. B: the original baseline as a control. C: winners, best free model."""
        judge = {"judge": self.winners["overrides"].get("judge", {})}
        winners = self.current_config()
        setups = {"A-all-winners": winners}
        setups["B-control-baseline"] = apply_overrides(self.baseline, judge)
        free = self.winners.get("best_free_generation")
        if free:
            setups["C-best-free-generator"] = apply_overrides(winners, {"generation": free})
        for extra in extra_setups or []:
            setups[extra["name"]] = apply_overrides(winners, extra["overrides"])
        unique: dict[str, dict] = {}
        for name, config in setups.items():
            if config not in unique.values():
                unique[name] = config
        return unique

    def score_final_anchor(self, name: str, anchor_id: str, config: dict) -> dict:
        """Answer and score one anchor's held-back questions for one final setup."""
        self.ensure_index(anchor_id, config)
        questions = load_golden(anchor_id, "test")
        run_dir = self.run_dir / anchor_id
        answers = answer_questions(anchor_id, questions, config, self.store, self.budget, run_dir)
        key = answer_key(anchor_id, config)
        score_path = run_dir / f"{key}.scores-{short_hash(config['judge'])}.jsonl"
        scores = score_answers(answers, config["judge"], self.budget, score_path)
        run_name = f"final-{name}-{anchor_id}"
        return {
            "anchor": anchor_id,
            "run_name": run_name,
            "run_key": short_hash([run_name, key, short_hash(config["judge"])]),
            "answers": answers,
            "scores": scores,
            "metrics": summarize(answers, scores),
            "answers_file": run_dir / f"{key}.answers.jsonl",
            "scores_file": score_path,
        }

    def run_final_setup(self, name: str, config: dict, experiment: str) -> dict:
        """Score one setup on all anchors, then log a parent run with one child per anchor."""
        parts = [self.score_final_anchor(name, anchor, config) for anchor in load_anchors()]
        combined = summarize(
            [row for part in parts for row in part["answers"]],
            [row for part in parts for row in part["scores"]],
        )
        for part in parts:
            self.report(part["run_name"], part["metrics"])
            self.record(
                {
                    "stage": "final",
                    "anchor": part["anchor"],
                    "setting": "final",
                    "variant": name,
                    "run_key": part["run_key"],
                    "split": "test",
                    "answers_path": part["answers_file"].relative_to(self.results).as_posix(),
                    "scores_path": part["scores_file"].relative_to(self.results).as_posix(),
                    "metrics": part["metrics"],
                    "generation": config["generation"],
                    "winner": False,
                }
            )
        self.report(f"final-{name}", combined)

        parent_key = short_hash([part["run_key"] for part in parts])
        experiment_id = get_experiment(experiment)
        found = MlflowClient().search_runs(
            [experiment_id], f"tags.run_key = '{parent_key}' and attributes.status = 'FINISHED'"
        )
        if found:
            return {
                "name": name,
                "config": config,
                "metrics": combined,
                "run_id": found[0].info.run_id,
            }
        tags = {"stage": "final", "split": "test", "setup": name}
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"final-{name}") as parent:
            mlflow.log_params(flatten(config) | {"split": "test", "setup": name})
            mlflow.set_tags(tags | {"run_key": parent_key, "primary_metric": PRIMARY_METRIC})
            mlflow.log_metrics({key: value for key, value in combined.items() if value is not None})
            for part in parts:
                log_run(
                    experiment,
                    part["run_name"],
                    part["run_key"],
                    config,
                    {"anchor": part["anchor"], "split": "test", "setup": name},
                    tags | {"anchor": part["anchor"]},
                    part["metrics"],
                    [
                        write_run_output(
                            self.outputs_dir / f"{part['run_name']}.jsonl",
                            part["answers"],
                            part["scores"],
                        )
                    ],
                    nested=True,
                )
            return {
                "name": name,
                "config": config,
                "metrics": combined,
                "run_id": parent.info.run_id,
            }

    def run_final(self, final: dict) -> dict:
        print("Final comparison on held-back questions", flush=True)
        setups = self.final_setups(final.get("extra_setups"))
        results = [
            self.run_final_setup(name, config, final["experiment"])
            for name, config in setups.items()
        ]
        best = pick_winner(results)
        served = choose_serving(results, best, final.get("serving_margin", 0.0))
        mark_winner(best["run_id"], "Best setup on the held-back questions.")
        MlflowClient().set_tag(served["run_id"], "serving", "true")
        (self.results / "best_run.json").write_text(
            json.dumps({"best": best, "serving": served, "all": results}, indent=2), "utf-8"
        )
        config = {key: value for key, value in served["config"].items() if key != "judge"}
        header = f"# Written by src/evaluate.py. Served setup: {served['name']}.\n"
        (CONFIGS / "serving.yaml").write_text(header + yaml.safe_dump(config), "utf-8")
        print(f"Best setup: {best['name']}. Served setup: {served['name']}.", flush=True)
        return best

    def write_reports(self) -> None:
        """Write the tables and side-by-side answers used in the README and report."""
        rows = [
            {
                "stage": run["stage"],
                "anchor": run["anchor"],
                "setting": run["setting"],
                "variant": run["variant"],
                "split": run["split"],
                "winner": run["winner"],
                **{name: run["metrics"].get(name) for name in RUN_METRICS},
            }
            for run in self.index
        ]
        if not rows:
            return
        with open(self.results / "summary.csv", "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        lines = ["# Same question, different settings", ""]
        seen = set()
        for run in self.index:
            group = (run["stage"], run["setting"])
            if group in seen or run["stage"] == "final":
                continue
            seen.add(group)
            lines += [f"## Stage {run['stage']}: {run['setting']} on {run['anchor']}", ""]
            runs = [item for item in self.index if (item["stage"], item["setting"]) == group]
            answers = [load_rows(self.results / item["answers_path"]) for item in runs]
            scores = [load_rows(self.results / item["scores_path"]) for item in runs]
            question_id = most_changed_question(answers, scores)
            lines += [f"**Question:** {answers[0][question_id]['question']}", ""]
            for item, answer_rows, score_rows in zip(runs, answers, scores, strict=True):
                mark = " (chosen)" if item["winner"] else ""
                faithfulness = score_rows[question_id].get("faithfulness")
                shown = "no claims made" if faithfulness is None else f"{faithfulness:.2f}"
                lines += [
                    f"### {item['variant']}{mark}: faithfulness {shown}",
                    "",
                    answer_rows[question_id]["answer"].strip(),
                    "",
                ]
        (self.results / "side_by_side.md").write_text("\n".join(lines), "utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=["dev", "test"], help="Default is dev, then test.")
    parser.add_argument("--stage", type=int, help="Run dev stages up to and including this one.")
    args = parser.parse_args()

    os.chdir(ROOT)
    plan = load_yaml(CONFIGS / "stages.yaml")
    mlflow.set_tracking_uri(TRACKING_URI)
    evaluation = Evaluation(PineconeStore())
    try:
        if args.split != "test":
            evaluation.reset_winners()
            for stage in plan["stages"]:
                if args.stage is None or stage["id"] <= args.stage:
                    evaluation.run_stage(stage)
        if args.split != "dev" and args.stage is None:
            evaluation.run_final(plan["final"])
    finally:
        evaluation.write_reports()
        database = ROOT / "mlflow.db"
        if database.exists():
            make_artifact_paths_relative(database)
        print(f"Estimated spend so far: ${evaluation.budget.spent:.3f}")


if __name__ == "__main__":
    main()
