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

## Milestone 1: reflections

### Abdullah Irfan

I built paper collection, data preparation, chunking and index building, and I set up the project and README. I learned that real PDF data needs careful checks. One bad rule left about 10,000 words of extra text in a paper, and we only found it by comparing word counts.

### Muhammad Fahad

I built retrieval, the prompt options, answer generation and the FastAPI service, and I set up CI. I learned that a bad answer is not always the model's fault. Sometimes the model refuses because retrieval did not find the right passage, so both parts need to be tested.

### Ateeb

I built the golden questions, the judge, the staged evaluation with MLflow, and the report. I learned that the judge has a big effect on the scores. The same answers scored from 0.78 to 0.96 with different judges, so one judge must score all the runs we compare.
