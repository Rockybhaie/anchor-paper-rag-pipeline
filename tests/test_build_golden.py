import random

import pytest

from src.build_golden import assign_splits, is_readable, is_usable, replace_questions, too_similar

ANSWER = "It updates a memory module during inference so that context is kept over time."


def question(number, kind="single"):
    return {"question": f"Question {number}?", "type": kind}


def test_is_usable_accepts_a_standalone_question():
    item = {"question": "How does Titans update its memory module at test time?", "answer": ANSWER}
    assert is_usable(item)


@pytest.mark.parametrize(
    "text",
    [
        "What does the excerpt say about memory modules?",
        "Why did the authors choose this memory design?",
        "What limitation does this paper identify in Titans?",
        "How does the passage describe surprise-based updates?",
        "How does the method ensure numerical stability over long horizons?",
        "How does the proposed approach keep the best energy found so far?",
        "How do routing and retrieval differ between the first work and the second work?",
        "How does random initialization contribute to the model's performance?",
        "Short?",
        "How does Titans update memory without a question mark",
    ],
)
def test_is_usable_rejects_questions_that_do_not_stand_alone(text):
    assert not is_usable({"question": text, "answer": ANSWER})


def test_is_usable_rejects_a_short_answer():
    assert not is_usable({"question": "How does Titans update its memory?", "answer": "It learns."})


def test_too_similar_catches_near_duplicates():
    existing = ["How does Titans update its memory module at test time?"]
    assert too_similar("How does Titans update the memory module at test time?", existing)
    assert not too_similar("Why does surprise gating reduce memory writes in MIRAS?", existing)


def test_is_readable_rejects_maths_and_tables():
    assert is_readable("Memory modules store context across long sequences during inference.")
    assert not is_readable("0.91 0.82 | 1.33 (2) 4.5 = 7.1 + 3.2 | 0.1 0.2 [3] [4]")
    assert not is_readable("")


def test_assign_splits_gives_15_dev_and_15_test_with_comparisons_in_both():
    questions = [question(i) for i in range(29)] + [question(i, "cross") for i in range(7)]

    result = assign_splits(questions, random.Random(1))

    for split, total, cross in (("dev", 15, 3), ("test", 15, 3), ("spare", 6, 1)):
        group = [q for q in result if q["split"] == split]
        assert len(group) == total
        assert sum(q["type"] == "cross" for q in group) == cross
    assert len({q["id"] for q in result}) == 36


def test_assign_splits_is_repeatable():
    def build():
        questions = [question(i) for i in range(29)] + [question(i, "cross") for i in range(7)]
        result = assign_splits(questions, random.Random(42))
        return [(q["id"], q["question"], q["split"]) for q in result]

    assert build() == build()


def test_replace_questions_promotes_a_spare_of_the_same_type():
    questions = [
        {"id": "q01", "split": "dev", "type": "single"},
        {"id": "q02", "split": "test", "type": "cross"},
        {"id": "q03", "split": "spare", "type": "cross"},
        {"id": "q04", "split": "spare", "type": "single"},
    ]

    result = {q["id"]: q["split"] for q in replace_questions(questions, ["q01"])}

    assert result == {"q02": "test", "q03": "spare", "q04": "dev"}


def test_replace_questions_rejects_unknown_ids():
    with pytest.raises(ValueError, match="q99"):
        replace_questions([{"id": "q01", "split": "dev", "type": "single"}], ["q99"])
