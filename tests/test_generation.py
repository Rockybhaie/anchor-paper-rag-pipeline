import pytest

from src.config import PROMPTS
from src.generation import build_messages, extract_answer, format_context, is_refusal, read_prompt

PASSAGES = [{"title": "Titans", "year": 2025, "text": "Memory is updated at test time."}]


def test_format_context_numbers_sources():
    assert "[Source 1: Titans (2025)]" in format_context(PASSAGES)


def test_original_prompt_has_no_system_message():
    messages = build_messages("p0", "What is updated?", PASSAGES)
    assert [message["role"] for message in messages] == ["user"]
    assert "Memory is updated" in messages[0]["content"]
    assert "What is updated?" in messages[0]["content"]


@pytest.mark.parametrize("prompt", ["p1", "p2", "p3", "p4"])
def test_other_prompts_use_a_system_instruction(prompt):
    messages = build_messages(prompt, "What is updated?", PASSAGES)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "{context}" not in messages[1]["content"]
    assert "{question}" not in messages[1]["content"]


def test_every_prompt_file_contains_both_placeholders():
    files = sorted((PROMPTS / "generation").glob("*.txt"))
    assert [path.stem for path in files] == ["p0", "p1", "p2", "p3", "p4"]
    for path in files:
        _, user = read_prompt(path)
        assert "{context}" in user and "{question}" in user


def test_extract_answer_removes_reasoning():
    assert extract_answer("[Reasoning]\nThinking.\n\n[Answer]\nThe result.") == "The result."
    assert extract_answer("Just an answer.") == "Just an answer."


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("The excerpts do not contain enough information to answer.", True),
        ("Memory is updated at test time (Source 1).", False),
    ],
)
def test_is_refusal(answer, expected):
    assert is_refusal(answer) is expected
