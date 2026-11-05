"""Download the papers in each manifest, validate them, and save clean text.

This step is separate from building the index. Example: python -m src.prepare_data --anchor titans
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import pymupdf
import requests

from src.config import ANCHOR_DATA, PROCESSED, RAW, load_anchors

MIN_WORDS = 500
MIN_PRINTABLE_RATIO = 0.85
# Papers with long appendices can start their references well before the middle of the text.
MIN_KEPT_AFTER_REFERENCES = 0.25
MIN_PAPERS_PER_ANCHOR = 5
USER_AGENT = "anchor-rag-course-project"

REFERENCE_HEADING = re.compile(
    r"\n[ \t]*(?:\d+\.?[ \t]*)?(references|bibliography)[ \t]*\n", re.IGNORECASE
)


def extract_text(pdf_path: Path) -> str:
    with pymupdf.open(pdf_path) as document:
        return "".join(page.get_text() for page in document)


def remove_references(text: str) -> str:
    """Cut from the last references heading, unless that would remove most of the paper."""
    matches = list(REFERENCE_HEADING.finditer(text))
    if matches and matches[-1].start() >= len(text) * MIN_KEPT_AFTER_REFERENCES:
        return text[: matches[-1].start()]
    return text


def clean_text(text: str) -> str:
    """Remove references, page numbers, divider lines and extra blank lines."""
    text = remove_references(text)
    text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
    text = re.sub(r"\n[-_=]{3,}\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_pdf_bytes(data: bytes) -> list[str]:
    return [] if data.startswith(b"%PDF-") else ["file is not a PDF"]


def validate_text(text: str) -> list[str]:
    """Return the problems that make extracted text unusable. An empty list means it is valid."""
    if not text.strip():
        return ["no text extracted"]
    problems = []
    words = len(text.split())
    if words < MIN_WORDS:
        problems.append(f"only {words} words, minimum is {MIN_WORDS}")
    printable = sum(1 for character in text if character.isprintable() or character.isspace())
    if printable / len(text) < MIN_PRINTABLE_RATIO:
        problems.append("too many unreadable characters")
    return problems


def download_pdf(urls: list[str], path: Path) -> bytes:
    """Return the PDF from disk, or from the first URL in the list that works."""
    if path.exists():
        return path.read_bytes()
    errors = []
    for url in urls:
        try:
            response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120)
            response.raise_for_status()
        except requests.RequestException as error:
            errors.append(f"{url}: {error}")
            continue
        if validate_pdf_bytes(response.content):
            errors.append(f"{url}: file is not a PDF")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        time.sleep(3)
        return response.content
    raise OSError("; ".join(errors))


def prepare_paper(paper: dict, pdf_path: Path, output_path: Path) -> dict:
    """Process one paper and return its report row, including why it was skipped if it was."""
    row = {"id": paper["id"], "title": paper["title"], "status": "kept", "warnings": []}
    try:
        data = download_pdf(paper["urls"], pdf_path)
    except OSError as error:
        return row | {"status": "skipped", "reason": f"download failed: {error}"}

    problems = validate_pdf_bytes(data)
    if problems:
        return row | {"status": "skipped", "reason": "; ".join(problems)}
    if paper.get("sha256") and hashlib.sha256(data).hexdigest() != paper["sha256"]:
        # arXiv can regenerate a PDF of the same version, so this is recorded but not fatal.
        row["warnings"].append("file hash differs from the manifest")

    try:
        text = clean_text(extract_text(pdf_path))
    except (RuntimeError, ValueError) as error:
        return row | {"status": "skipped", "reason": f"could not read PDF: {error}"}
    problems = validate_text(text)
    if problems:
        return row | {"status": "skipped", "reason": "; ".join(problems)}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text, encoding="utf-8")
    return row | {"words": len(text.split())}


def prepare_anchor(anchor_id: str) -> dict:
    """Prepare all papers for one anchor and save a report. Fails if too few papers survive."""
    manifest = json.loads((ANCHOR_DATA / anchor_id / "manifest.json").read_text(encoding="utf-8"))
    rows = [
        prepare_paper(
            paper,
            RAW / anchor_id / "pdf" / f"{paper['id']}.pdf",
            PROCESSED / anchor_id / f"{paper['id']}.txt",
        )
        for paper in manifest["papers"]
    ]
    kept = [row for row in rows if row["status"] == "kept"]
    report = {"anchor": anchor_id, "kept": len(kept), "total": len(rows), "papers": rows}
    report_path = PROCESSED / anchor_id / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{anchor_id}: kept {len(kept)} of {len(rows)} papers")
    for row in rows:
        if row["status"] == "skipped":
            print(f"  skipped {row['id']}: {row['reason']}")
        for warning in row["warnings"]:
            print(f"  warning {row['id']}: {warning}")
    if len(kept) < MIN_PAPERS_PER_ANCHOR:
        raise ValueError(f"{anchor_id}: only {len(kept)} usable papers")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    anchors = sorted(load_anchors())
    parser.add_argument("--anchor", choices=anchors, help="Prepare one anchor. Default is all.")
    args = parser.parse_args()
    try:
        for anchor_id in [args.anchor] if args.anchor else anchors:
            prepare_anchor(anchor_id)
    except ValueError as error:
        sys.exit(str(error))


if __name__ == "__main__":
    main()
