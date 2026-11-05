import json

import pymupdf
import pytest

from src import prepare_data
from src.prepare_data import (
    clean_text,
    prepare_paper,
    remove_references,
    validate_pdf_bytes,
    validate_text,
)

BODY = "Nested learning treats a model as several levels of optimization. " * 60


def make_pdf(path, text):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_textbox(pymupdf.Rect(40, 40, 560, 800), text, fontsize=6)
    document.save(path)
    document.close()


def test_clean_text_removes_references_page_numbers_and_dividers():
    raw = f"Title\n\nIntro\n12\nMore\n-----\nBody\n\n\n\nEnd {BODY}\nReferences\n[1] A paper."
    cleaned = clean_text(raw)
    assert "[1] A paper" not in cleaned
    assert "\n12\n" not in cleaned
    assert "-----" not in cleaned
    assert "\n\n\n" not in cleaned


def test_remove_references_uses_the_last_heading():
    text = f"Intro\nReferences\nThis early heading is a false match.\n{BODY}\nReferences\n[1] A."
    result = remove_references(text)
    assert "false match" in result
    assert "[1] A." not in result


def test_remove_references_keeps_text_when_the_heading_comes_too_early():
    text = f"Intro\nReferences\n{BODY}"
    assert remove_references(text) == text


def test_remove_references_handles_a_long_appendix_after_the_references():
    body = "x" * 3600
    text = f"{body}\nReferences\n[1] A cited paper.\n{'y' * 6400}"
    assert remove_references(text) == body


def test_validate_text_accepts_a_real_paper():
    assert validate_text(BODY) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [("", "no text"), ("   \n ", "no text"), ("too short", "only 2 words")],
)
def test_validate_text_rejects_unusable_text(text, expected):
    assert expected in validate_text(text)[0]


def test_validate_text_rejects_unreadable_characters():
    text = ("word " * 600) + ("\x00" * 2000)
    assert "unreadable" in validate_text(text)[0]


def test_validate_pdf_bytes():
    assert validate_pdf_bytes(b"%PDF-1.7 content") == []
    assert validate_pdf_bytes(b"<html>Not a paper</html>") == ["file is not a PDF"]


def test_prepare_paper_keeps_a_valid_pdf_and_warns_on_a_changed_hash(tmp_path):
    pdf = tmp_path / "paper.pdf"
    make_pdf(pdf, BODY)
    output = tmp_path / "out" / "paper.txt"
    paper = {"id": "paper", "title": "T", "urls": [], "sha256": "different"}

    row = prepare_paper(paper, pdf, output)

    assert row["status"] == "kept"
    assert row["warnings"] == ["file hash differs from the manifest"]
    assert "Nested learning" in output.read_text(encoding="utf-8")


def test_prepare_paper_records_why_a_download_failed(tmp_path, monkeypatch):
    def fail(urls, path):
        raise OSError("site returned 403")

    monkeypatch.setattr(prepare_data, "download_pdf", fail)
    paper = {"id": "paper", "title": "T", "urls": ["https://example.com/p.pdf"]}

    row = prepare_paper(paper, tmp_path / "p.pdf", tmp_path / "p.txt")

    assert row["status"] == "skipped"
    assert "403" in row["reason"]
    assert not (tmp_path / "p.txt").exists()


def test_prepare_anchor_saves_a_report_and_fails_with_too_few_papers(tmp_path, monkeypatch):
    anchors = tmp_path / "anchors" / "demo"
    anchors.mkdir(parents=True)
    papers = [{"id": "a", "title": "A", "urls": []}]
    (anchors / "manifest.json").write_text(json.dumps({"papers": papers}), encoding="utf-8")
    monkeypatch.setattr(prepare_data, "ANCHOR_DATA", tmp_path / "anchors")
    monkeypatch.setattr(prepare_data, "RAW", tmp_path / "raw")
    monkeypatch.setattr(prepare_data, "PROCESSED", tmp_path / "processed")
    raw = tmp_path / "raw" / "demo" / "pdf"
    raw.mkdir(parents=True)
    make_pdf(raw / "a.pdf", BODY)

    with pytest.raises(ValueError, match="only 1 usable"):
        prepare_data.prepare_anchor("demo")

    report = json.loads((tmp_path / "processed" / "demo" / "report.json").read_text("utf-8"))
    assert report["kept"] == 1
