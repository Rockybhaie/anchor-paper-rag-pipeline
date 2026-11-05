# Anchor Paper RAG: Milestone 1 Report

<p class="byline">MLOps course project. Abdullah Irfan (29266), Muhammad Fahad (29264), Ateeb (29010).</p>

## Summary

We built a question-answering system over four research papers and the papers that cite them. It answers only from retrieved passages and names its sources. Following the course's update for RAG projects, we evaluated its inference instead of training cycles, changing prompts, models, parameters, temperature and judges across 57 MLflow runs.

**We serve hybrid search through Pinecone with the free `gemini-3.5-flash-lite` model. It scored 0.978 faithfulness on 60 held-back questions, against 0.928 for the earlier project's setup, and refused 5% of questions against 18%.** One keyword-only setup scored slightly higher at 0.985, and section 5 explains why we did not serve it.

## 1. System

<div class="flow">
<div><b>Prepare</b>Download 60 papers from frozen lists, validate, clean</div><span>&rarr;</span>
<div><b>Build index</b>Split, embed, upload to Pinecone, save versioned passages</div><span>&rarr;</span>
<div><b>Find passages</b>Search by keywords and meaning, keep the best 8</div><span>&rarr;</span>
<div><b>Answer</b>LLM writes from those passages only, citing them</div><span>&rarr;</span>
<div><b>Serve</b>FastAPI: <code>/predict</code>, <code>/explain</code>, <code>/docs</code></div>
</div>

Building the search index takes the place of training, and the saved index is the artifact the API loads. Faithfulness and the other scores are computed by a separate evaluation script, not at request time.

## 2. Data

Each anchor has 15 papers: the anchor and 14 papers that cite it. No paper appears in two corpora.

{{corpus_table}}

Anchor 1 keeps the earlier project's hand-chosen papers. For the other three, a script listed citing papers from Semantic Scholar and an LLM scored their abstracts. We then ranked citations that Semantic Scholar flags as influential first, because the LLM rated most papers highly. The team approved the final lists, and full papers are not stored in the repository.

## 3. Evaluation design

**Metric.** Faithfulness is the share of an answer's claims that the retrieved passages support, and it is the only metric used to choose. We also log relevancy, whether the question's source paper was retrieved, refusals, latency and cost.

**Questions.** We wrote 30 new questions per anchor, drawn from across each paper, with about 1 in 5 comparing two papers. 15 per anchor were used to choose settings and 15 were held back. The team removed 13 questions that did not stand alone.

**Staged runs.** Testing every combination would need about 38,000 runs, so each stage tries a few settings on one anchor and passes its winners on. All chosen settings are then checked on the held-back questions of all four anchors.

**Choosing a winner.** Highest faithfulness wins. Near ties within 0.01 go to higher relevancy, then lower latency. Runs that refuse more than 34% of questions, or answer fewer than 80%, cannot win.

## 4. Results by setting

**Judge.** Scores from different judges cannot be compared, so one judge must score every later run. A high score alone does not show a judge is good, so we scored each answer against its real passages and against unrelated ones, and chose the largest gap. The same answers scored from 0.78 to 0.96 depending on the judge.

{{judge_table}}

{{stage_charts}}

### What changed the results most

- **Prompts.** Faithfulness ranged from 0.935 to 0.990. The short extractive prompt P3 scored highest, but its relevancy was lower, so the close tie went to P1, strict instructions with citations.
- **Chunking.** Larger recursive chunks (800 characters) beat the earlier project's 400. Semantic chunking tied with recursive 400 once we removed a 2,500-character limit on what the judge could read. That limit appears to have favoured short chunks in the earlier project, though the questions and judge also changed.
- **Models.** The free Gemini model was within 0.01 of `gpt-4o-mini` here, and `gpt-oss-20b` trailed at 0.922.
- **Temperature.** Lower was generally better, but not smoothly: 0.7 scored higher than 0.4, which suggests noise on 15 questions.
- **Embeddings.** `text-embedding-3-small` scored 0.950 and the free local `all-MiniLM-L6-v2` 0.937.

## 5. Final results on held-back questions

Setups D and E were added after seeing the first results for A, B and C, once we noticed keyword search never queries Pinecone. Their comparison is therefore slightly less clean than a fully planned one.

{{final_table}}

<p class="note">B is the earlier project's setup: recursive 400/80 chunks, 5 passages, prompt P0, temperature 0.1. All others use recursive 800/120 chunks, 8 passages, prompt P1, temperature 0.0. All were scored by the same judge.</p>

{{anchor_table}}

**Best run and why we serve D.** Setup C has the top score, but D is only 0.007 behind, too small a gap to treat as real on 60 questions. We serve D for three reasons:

- It searches by meaning as well as keywords, as our proposal described. Keyword-only retrieval never queries Pinecone when answering.
- The test questions favour keyword search, because our question rules told the generator to name the method being asked about. Real users paraphrase.
- It refuses least and is the most consistent. It refuses 5% of questions against 10% for C, and its faithfulness ranges from 0.976 to 0.979 across anchors, against 0.967 to 0.993 for C.

**Trade-offs.** Compared with the earlier project's setup, D is more faithful and refuses much less, but its relevancy is lower (0.702 against 0.755 on answered questions). It is also slower, at about 10 seconds against 3.6, mostly because the free model responds slowly. Setup E is the alternative if speed matters more: it takes 3.9 seconds and scores 0.962.

## 6. How settings change the answers

These are the questions each setting changed the most, so they show the largest effects, not typical ones. Each compares the chosen setting with the most different answer. Answers are shortened, and more are in `results/side_by_side.md`.

{{example:prompt}}

{{example:temperature}}

{{example:model}}

## 7. Limits

- **Small samples.** Each stage used 15 questions on one anchor. Setup A combined every stage winner, yet changing only its generator gave a higher score, so stage winners do not simply add up.
- **LLM-written questions.** They test finding and using what is in the corpus, not what real researchers would ask.
- **LLM judges.** Faithfulness and relevancy are judgements that vary slightly between runs, and `gpt-4o-mini` judges its own answers in setups A, B and E.
- **What faithfulness misses.** An answer can be faithful to the wrong passages, which is why we also log whether the right paper was found. Refusals are left out of faithfulness and tracked separately.
- **Relevancy has its own limits.** A refusal repeats the question's words, so it can score well, which is why the table also shows relevancy on answered questions. In Stage 3, stricter prompts lowered it, from 0.730 for the original prompt to 0.715 for P1, and we chose faithfulness as the main metric knowing this.
- **Latency.** Times were measured on a laptop CPU, with a free model that varied from about 3 to 14 seconds per answer across anchors.

Changes from the proposal, including fixing four anchors for this milestone, are listed in `README.md`.

## 8. MLflow

Every run records its settings, metrics and per-question outputs. The history is committed in `mlflow.db` and `mlartifacts/`, so anyone can open it without API keys.

We rebuilt the MLflow history from saved answers after fixing measurement bugs, so its timestamps and durations reflect that logging step, not the original model calls. Latency appears as its own metric on each run.

{{shot:mlflow_experiments|One MLflow experiment for each stage, plus the final held-back test.}}

{{shot:mlflow_final_runs|Final comparison runs. Each setup has one child run per anchor.}}

{{shot:mlflow_compare|Comparing the five prompt runs from Stage 3 in MLflow.}}

{{shot:mlflow_run_detail|One run's metrics. Its parameters are further down this page, and its saved outputs are under Artifacts.}}
