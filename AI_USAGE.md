# AI usage

## Our work

This project builds on our own earlier project, which supplied anchor 1's paper selection, the retrieval design, the chunking settings, the first prompts and the judge approach. We wrote the idea and proposal for this project ourselves.

The team made the main decisions:

- **Scope:** four fixed anchor papers, with 14 citing papers for each.
- **Index:** using Pinecone for the search index.
- **Evaluation design:** testing a few settings on each anchor and passing the winners on, instead of running every setting on every anchor.
- **What to compare:** prompt options, judge models, and OpenAI against free models.
- **Deliverables:** a results report with MLflow screenshots.
- **Work split:** which member owns which part of the system.

The team also approved the citing papers for each anchor, approved the removal of 13 golden questions that did not stand alone, and tested and reviewed the system before submission.

## Tools used

| Tool | Used for |
|---|---|
| Claude Code (Anthropic, Claude Opus 5.5) | Planning the milestone with the team, drafting most of the code, tests, configuration and documentation, running the pipeline and evaluation, debugging, and drafting the report |

Claude Code wrote most of the first drafts of the code in `src/`, `tests/`, `tools/` and `configs/`, along with `README.md`, the report, the member reflections in `CONTRIBUTION.md`, and this file's first draft.

## Models used inside the project

These are part of the system, separate from the coding assistant above:

- `gpt-4o-mini` scored paper abstracts, wrote the golden questions, and acted as the evaluation judge.
- `gpt-4o-mini`, `gpt-5-nano`, `gemini-3.5-flash-lite` and `gpt-oss-20b` were compared as answer generators, and as judges.

## Group confirmation

We have read and reviewed all AI-suggested code, tested it, and understand it. Each of us can explain any part of the system, including parts we did not personally write.
