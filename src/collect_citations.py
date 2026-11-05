"""Choose the papers for each anchor and write the frozen manifest. Run once, not on a clean clone.

Steps:
  fetch     Get the citing papers from Semantic Scholar. No key needed.
  score     Score each abstract on the five criteria from the earlier project.
  finalize  Check PDFs and page counts, choose the top papers and write manifest.json.
  seed      Write the manifest for an anchor whose papers were chosen by hand.

Example: python -m src.collect_citations fetch --anchor titans
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import xml.etree.ElementTree as ElementTree
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pymupdf
import requests

from src.config import ANCHOR_DATA, RAW, load_anchors, load_yaml
from src.llm import chat, estimate_cost, get_client

S2_URL = "https://api.semanticscholar.org/graph/v1/paper/arXiv:{arxiv}/citations"
S2_FIELDS = "title,year,abstract,authors,externalIds,citationCount,isInfluential"
ARXIV_API = "https://export.arxiv.org/api/query?id_list={ids}&max_results={count}"
OLD_REPO_PDFS = (
    "https://raw.githubusercontent.com/Rockybhaie/"
    "Tracking-the-Evolution-of-Nested-Learning-via-RAG/1f41c68/"
    "Part%201%20Preprocessing/Dataset/Selected%20Papers/{file}"
)
USER_AGENT = "anchor-rag-course-project"
MIN_PAGES = 5
MAX_PAGES = 60
# Keeps each corpus varied. Close to the 5, 5 and 4 split of the hand-chosen anchor 1 corpus.
CATEGORY_MIX = {"Core Extension": 7, "Application": 4, "Related Theory": 3}
SCORE_MODEL = "gpt-4o-mini"
SCORE_PROMPT = """You are helping choose papers for a corpus about how later work
responded to one anchor paper.

Anchor paper: {anchor}

Citing paper title: {title}
Citing paper abstract: {abstract}

Score the citing paper from 1 to 5 on each criterion:
- relevance: how closely it relates to the anchor paper's core concepts
- depth: how much conceptual or technical substance it offers
- density: how much useful information it is likely to contain per page

Also choose one category:
- Core Extension: directly extends the anchor's method
- Application: applies the anchor's ideas in a new setting
- Related Theory: gives related theory or a comparison
- Weak Citation: probably only mentions the anchor in passing

Return only JSON in this format:
{{"relevance": 1, "depth": 1, "density": 1, "category": "Application",
"reason": "one short sentence"}}"""


def _request(url: str, tries: int = 7, **kwargs) -> requests.Response:
    headers = {"User-Agent": USER_AGENT}
    if os.environ.get("SEMANTIC_SCHOLAR_API_KEY") and "semanticscholar" in url:
        headers["x-api-key"] = os.environ["SEMANTIC_SCHOLAR_API_KEY"]
    wait = 8.0
    for attempt in range(tries):
        response = requests.get(url, headers=headers, timeout=60, **kwargs)
        if response.status_code != 429 or attempt == tries - 1:
            response.raise_for_status()
            return response
        time.sleep(wait)
        wait = min(wait * 2, 60.0)
    raise RuntimeError(f"Could not fetch {url}")


def fetch_candidates(anchor_id: str) -> list[dict]:
    """Return citing papers that have an arXiv copy and an abstract, and save the list at once."""
    arxiv = load_anchors()[anchor_id]["arxiv"]
    rows, offset = [], 0
    while True:
        page = _request(
            S2_URL.format(arxiv=arxiv),
            params={"limit": 500, "offset": offset, "fields": S2_FIELDS},
        ).json()
        rows.extend(page.get("data", []))
        if "next" not in page or not page.get("data"):
            break
        offset = page["next"]
        time.sleep(3)

    candidates = {}
    for row in rows:
        paper = row["citingPaper"]
        arxiv_id = (paper.get("externalIds") or {}).get("ArXiv")
        if arxiv_id and paper.get("abstract") and arxiv_id != arxiv:
            candidates[arxiv_id] = {
                "arxiv": arxiv_id,
                "title": paper["title"],
                "year": paper.get("year"),
                "authors": ", ".join(a["name"] for a in (paper.get("authors") or [])[:6]),
                "abstract": paper["abstract"],
                "influential": bool(row.get("isInfluential")),
                "citation_count": paper.get("citationCount") or 0,
            }

    result = sorted(candidates.values(), key=lambda c: c["arxiv"])
    path = RAW / anchor_id / "candidates_with_abstracts.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{anchor_id}: {len(rows)} citing papers listed, {len(result)} with arXiv and abstract")
    return result


def parse_score(text: str) -> dict:
    """Read the model's JSON reply and add the total score."""
    cleaned = text.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    data["total"] = sum(int(data[key]) for key in ("relevance", "depth", "density"))
    return data


def score_candidates(anchor_id: str) -> None:
    """Score each abstract. Finished papers are skipped, so this can be restarted."""
    anchor = load_anchors()[anchor_id]
    candidates = json.loads(
        (RAW / anchor_id / "candidates_with_abstracts.json").read_text(encoding="utf-8")
    )
    path = RAW / anchor_id / "scores.json"
    scores = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    client = get_client("openai")
    cost = 0.0
    for index, candidate in enumerate(candidates):
        if candidate["arxiv"] in scores:
            continue
        prompt = SCORE_PROMPT.format(
            anchor=anchor["title"], title=candidate["title"], abstract=candidate["abstract"]
        )
        result = chat(
            "openai",
            SCORE_MODEL,
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=120,
            client=client,
        )
        cost += estimate_cost(SCORE_MODEL, result.input_tokens, result.output_tokens)
        try:
            scores[candidate["arxiv"]] = parse_score(result.text)
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as error:
            print(f"  could not read score for {candidate['arxiv']}: {error}")
        if index % 25 == 0:
            path.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    path.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    print(f"{anchor_id}: {len(scores)} scored, estimated cost ${cost:.3f}")


def latest_arxiv_versions(arxiv_ids: list[str]) -> dict[str, str]:
    """Return the current version of each paper, such as {"2501.00663": "v2"}."""
    versions: dict[str, str] = {}
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    for start in range(0, len(arxiv_ids), 50):
        batch = arxiv_ids[start : start + 50]
        url = ARXIV_API.format(ids=",".join(batch), count=len(batch))
        root = ElementTree.fromstring(_request(url).content)
        for entry in root.findall("atom:entry", namespace):
            entry_id = entry.findtext("atom:id", "", namespace).rsplit("/", 1)[-1]
            base, _, version = entry_id.rpartition("v")
            if base and version.isdigit():
                versions[base] = f"v{version}"
        time.sleep(3)
    return versions


def inspect_pdf(url: str, path: Path) -> dict | None:
    """Download a PDF once and return its hash and page count, or None if it is unusable."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            response = _request(url, tries=3)
        except requests.RequestException as error:
            print(f"  download failed {url}: {error}")
            return None
        if not response.content.startswith(b"%PDF-"):
            print(f"  not a PDF: {url}")
            return None
        path.write_bytes(response.content)
        time.sleep(3)
    data = path.read_bytes()
    with pymupdf.open(path) as document:
        return {"sha256": hashlib.sha256(data).hexdigest(), "pages": document.page_count}


def write_manifest(anchor_id: str, papers: list[dict]) -> None:
    path = ANCHOR_DATA / anchor_id / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"anchor": anchor_id, "papers": papers}, indent=2), encoding="utf-8")
    citing = sum(1 for p in papers if p["role"] == "citing")
    print(f"{anchor_id}: manifest written with the anchor and {citing} citing papers")


def title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", title.lower())[:40]


def titles_in_use(anchor_id: str) -> set[str]:
    """Return all anchors and papers chosen for earlier anchors, so each corpus is distinct."""
    anchors = load_anchors()
    used = {title_key(anchor["title"]) for anchor in anchors.values()}
    for other in anchors:
        if other == anchor_id:
            break
        path = ANCHOR_DATA / other / "manifest.json"
        if path.exists():
            papers = json.loads(path.read_text(encoding="utf-8"))["papers"]
            used |= {title_key(paper["title"]) for paper in papers}
    return used


def rank_candidates(candidates: list[dict], scores: dict) -> list[dict]:
    """Rank by real engagement first, since abstract scores alone rate most papers highly."""
    scored = (c | scores[c["arxiv"]] for c in candidates if c["arxiv"] in scores)
    return sorted(
        (c for c in scored if c["category"] != "Weak Citation"),
        key=lambda c: (c["influential"], c["total"], c["citation_count"]),
        reverse=True,
    )


def select_papers(
    ranked: list[dict], target: int, used: set[str], accept: Callable[[dict], dict | None]
) -> list[dict]:
    """Take the best papers in the category mix, then fill any gap from the remaining papers."""
    chosen: list[dict] = []
    per_category: Counter = Counter()
    tried: set[str] = set()
    for keep_mix in (True, False):
        for candidate in ranked:
            if len(chosen) == target:
                return chosen
            category = candidate["category"]
            if candidate["arxiv"] in tried or title_key(candidate["title"]) in used:
                continue
            if keep_mix and per_category[category] >= CATEGORY_MIX.get(category, 0):
                continue
            tried.add(candidate["arxiv"])
            paper = accept(candidate)
            if paper is not None:
                chosen.append(paper)
                per_category[category] += 1
    return chosen


def finalize(anchor_id: str) -> None:
    """Choose the papers for one anchor, check their PDFs and write the manifest."""
    anchor = load_anchors()[anchor_id]
    raw = RAW / anchor_id
    candidates = json.loads((raw / "candidates_with_abstracts.json").read_text(encoding="utf-8"))
    scores = json.loads((raw / "scores.json").read_text(encoding="utf-8"))
    ranked = rank_candidates(candidates, scores)

    def entry(paper_id: str, role: str, arxiv_id: str, extra: dict) -> dict | None:
        version = latest_arxiv_versions([arxiv_id]).get(arxiv_id, "")
        url = f"https://arxiv.org/pdf/{arxiv_id}{version}"
        info = inspect_pdf(url, raw / "pdf" / f"{paper_id}.pdf")
        if info is None:
            return None
        return {"id": paper_id, "role": role, "urls": [url], **info, **extra}

    anchor_entry = entry(
        "00_anchor",
        "anchor",
        anchor["arxiv"],
        {key: anchor[key] for key in ("title", "authors", "year")} | {"category": "Anchor"},
    )
    if anchor_entry is None:
        raise RuntimeError(f"Could not download the anchor paper for {anchor_id}.")

    fields = ("title", "authors", "year", "category", "total", "influential", "reason")

    def accept(candidate: dict) -> dict | None:
        paper = entry(
            candidate["arxiv"].replace(".", "_"),
            "citing",
            candidate["arxiv"],
            {key: candidate[key] for key in fields},
        )
        if paper is not None and not MIN_PAGES <= paper["pages"] <= MAX_PAGES:
            print(f"  skipped {candidate['arxiv']}: {paper['pages']} pages")
            return None
        return paper

    target = anchor["citing_papers"]
    chosen = select_papers(ranked, target, titles_in_use(anchor_id), accept)
    if len(chosen) < target:
        print(f"  warning: only {len(chosen)} of {target} citing papers found for {anchor_id}")
    write_manifest(anchor_id, [anchor_entry, *chosen])

    keep = ("arxiv", "title", "year", "total", "category", "influential", "citation_count")
    summary = [{key: c.get(key) for key in keep} for c in ranked]
    (ANCHOR_DATA / anchor_id / "candidates.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8"
    )


def seed(anchor_id: str) -> None:
    """Write a manifest from a hand-chosen list in seed.yaml."""
    papers = load_yaml(ANCHOR_DATA / anchor_id / "seed.yaml")["papers"]
    raw = RAW / anchor_id / "pdf"
    versions = latest_arxiv_versions([p["arxiv"] for p in papers if p.get("arxiv")])
    manifest = []
    for paper in papers:
        fallback = OLD_REPO_PDFS.format(file=paper.get("old_repo_file", f"{paper['id']}.pdf"))
        urls = [fallback]
        if paper.get("arxiv"):
            version = versions.get(paper["arxiv"], "")
            urls.insert(0, f"https://arxiv.org/pdf/{paper['arxiv']}{version}")
        info = inspect_pdf(urls[0], raw / f"{paper['id']}.pdf")
        if info is None:
            raise RuntimeError(f"Could not download {paper['id']}.")
        details = {key: paper[key] for key in ("title", "authors", "year", "category")}
        manifest.append({"id": paper["id"], "role": paper["role"], "urls": urls, **info, **details})
    write_manifest(anchor_id, manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=["fetch", "score", "finalize", "seed"])
    parser.add_argument("--anchor", required=True, choices=sorted(load_anchors()))
    args = parser.parse_args()
    {
        "fetch": fetch_candidates,
        "score": score_candidates,
        "finalize": finalize,
        "seed": seed,
    }[args.step](args.anchor)


if __name__ == "__main__":
    main()
