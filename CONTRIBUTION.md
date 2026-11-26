# Contributions

## Team

| Name | ERP ID |
|---|---|
| Abdullah Irfan | 29266 |
| Muhammad Fahad | 29264 |
| Ateeb | 29010 |

## Milestone 1: member-to-task table

| Member | Tasks | Main files |
|---|---|---|
| Abdullah Irfan | Paper collection and selection, data preparation and validation, chunking, embeddings, model connections, Pinecone storage, index building, project setup, README | `src/collect_citations.py`, `src/prepare_data.py`, `src/chunking.py`, `src/embeddings.py`, `src/llm.py`, `src/vector_store.py`, `src/build_index.py`, `src/config.py`, `configs/anchors.yaml`, `configs/baseline.yaml`, `data/anchors/`, `Makefile`, `README.md` and their tests |
| Muhammad Fahad | Retrieval, prompt options, answer generation, FastAPI service, CI | `src/retrieval.py`, `src/generation.py`, `src/rag.py`, `src/schemas.py`, `src/app.py`, `prompts/generation/`, `configs/serving.yaml`, `configs/free.yaml`, `.github/workflows/ci.yml` and their tests |
| Ateeb | Golden questions, judge, staged evaluation, MLflow tracking, results, report | `src/build_golden.py`, `src/judge.py`, `src/evaluate.py`, `configs/stages.yaml`, `prompts/judge/`, `prompts/golden/`, `data/golden/`, `results/`, `mlflow.db`, `mlartifacts/`, `reports/`, `tools/`, `AI_USAGE.md` and their tests |

## Milestone 2: member-to-task table

| Member | Tasks | Main files |
|---|---|---|
| Abdullah Irfan | Rebuilding the passages from Pinecone so the container needs no paper text, the Docker image, deployment on Render, runbook, deployment and Docker sections of the README, release tag | `src/build_index.py`, `src/vector_store.py`, index loading in `src/rag.py`, lighter imports in `src/embeddings.py` and `src/retrieval.py`, `data/index_manifests/`, `Dockerfile`, `.dockerignore`, `requirements-serve.txt`, `RUNBOOK.md`, `README.md` and their tests |
| Muhammad Fahad | Monitoring with request count and latency, the `/health` and `/stats` endpoints, the Docker build job in CI | `src/app.py`, `.github/workflows/ci.yml`, `tests/test_app.py` |
| Ateeb | Project Focus on LLM-as-a-Judge: live judging of served answers, the judge-against-people check logged in MLflow, the Project Focus section of the README | `src/live_judge.py`, `src/rag.py`, the judge hook in `src/app.py`, `src/judge_agreement.py`, `results/judge_agreement/`, `tests/test_live_judge.py`, `tests/test_judge_agreement.py` |

## Milestone 1: reflections

### Abdullah Irfan

I built paper collection, data preparation, chunking and index building, and I set up the project and README. I learned that real PDF data needs careful checks. One bad rule left about 10,000 words of extra text in a paper, and we only found it by comparing word counts.

### Muhammad Fahad

I built retrieval, the prompt options, answer generation and the FastAPI service, and I set up CI. I learned that a bad answer is not always the model's fault. Sometimes the model refuses because retrieval did not find the right passage, so both parts need to be tested.

### Ateeb

I built the golden questions, the judge, the staged evaluation with MLflow, and the report. I learned that the judge has a big effect on the scores. The same answers scored from 0.78 to 0.96 with different judges, so one judge must score all the runs we compare.

## Milestone 2: reflections

### Abdullah Irfan

I packaged the API in Docker and deployed it on Render. I learned that a container cannot rely on files that only exist on my laptop. Our passages were gitignored, so I stored them with their vectors in Pinecone and had the container rebuild and check them on start.

### Muhammad Fahad

I added the `/health` and `/stats` endpoints and the Docker build job in CI. I learned that one average can mislead: frequent health checks take about a millisecond and pull the overall latency down, so `/stats` also reports each endpoint on its own.

### Ateeb

I started our LLM-as-a-Judge focus: judging live answers after they are sent, and a check of the judge against our own labels. I learned that a sample needs planning: the judge supported almost every claim, so a random sample would contain almost no disagreements to learn from.
