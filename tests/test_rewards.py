"""Tests for verifiable math rewards."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.rl.rewards import (
    RewardBreakdown,
    answers_equivalent,
    batch_reward,
    extract_boxed_answer,
    extract_gsm8k_answer,
    reward,
    reward_breakdown,
)


# ============================================================
#  Extractors
# ============================================================
class TestExtractGSM8K:
    def test_basic_int(self):
        assert extract_gsm8k_answer("So the total is 42.\n#### 42") == "42"

    def test_negative(self):
        assert extract_gsm8k_answer("#### -7") == "-7"

    def test_decimal(self):
        assert extract_gsm8k_answer("#### 3.14") == "3.14"

    def test_with_extra_whitespace(self):
        assert extract_gsm8k_answer("####   42  ") == "42"

    def test_fallback_answer_is(self):
        text = "After working through it, the answer is 100."
        assert extract_gsm8k_answer(text) == "100"

    def test_fallback_takes_last(self):
        text = "I thought the answer is 5, but actually the answer is 12."
        assert extract_gsm8k_answer(text) == "12"

    def test_prefer_terminator_over_fallback(self):
        text = "the answer is 5\n#### 12"
        # `####` wins over fallback when both exist
        assert extract_gsm8k_answer(text) == "12"

    def test_no_match_returns_none(self):
        assert extract_gsm8k_answer("this has no number marker at all") is None

    def test_empty_string(self):
        assert extract_gsm8k_answer("") is None


class TestExtractBoxed:
    def test_basic(self):
        assert extract_boxed_answer("the answer is \\boxed{42}") == "42"

    def test_returns_none_when_missing(self):
        assert extract_boxed_answer("no box here") is None

    def test_handles_nested_braces(self):
        assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"

    def test_handles_double_nested(self):
        assert (
            extract_boxed_answer("\\boxed{\\frac{a^{2}}{b^{2}}}")
            == "\\frac{a^{2}}{b^{2}}"
        )

    def test_takes_last_box(self):
        text = "first \\boxed{1} then \\boxed{2}"
        assert extract_boxed_answer(text) == "2"

    def test_unbalanced_returns_what_we_found(self):
        # Unbalanced; the function returns the last successfully parsed box
        # (or None if no successful parse before the bad one).
        result = extract_boxed_answer("\\boxed{42")
        assert result is None  # never balanced → never succeeded

    def test_strips_whitespace(self):
        assert extract_boxed_answer("\\boxed{  42  }") == "42"

    def test_empty_box(self):
        assert extract_boxed_answer("\\boxed{}") == ""


# ============================================================
#  Equivalence
# ============================================================
class TestAnswersEquivalent:
    def test_exact_match(self):
        assert answers_equivalent("42", "42")

    def test_int_vs_decimal(self):
        assert answers_equivalent("42", "42.0")
        assert answers_equivalent("42.000", "42")

    def test_negatives(self):
        assert answers_equivalent("-7", "-7")
        assert answers_equivalent("-7.0", "-7")

    def test_different_numbers_not_equiv(self):
        assert not answers_equivalent("42", "43")

    def test_none_inputs(self):
        assert not answers_equivalent(None, "42")
        assert not answers_equivalent("42", None)
        assert not answers_equivalent(None, None)

    def test_empty_inputs(self):
        assert not answers_equivalent("", "42")
        assert not answers_equivalent("42", "")

    def test_fraction_equivalence(self):
        # 1/2 == 0.5 via SymPy fallback
        assert answers_equivalent("\\frac{1}{2}", "0.5")
        assert answers_equivalent("1/2", "0.5")

    def test_fraction_simplification(self):
        # 4/8 == 1/2
        assert answers_equivalent("4/8", "1/2")

    def test_unrelated_strings_not_equiv(self):
        assert not answers_equivalent("apple", "banana")

    def test_unparseable_dont_crash(self):
        # Garbled LaTeX should return False, not raise
        assert not answers_equivalent("\\frac{not", "42")

    def test_thousands_separator(self):
        # "1,234" should equal "1234"
        assert answers_equivalent("1,234", "1234")

    def test_pi_constant(self):
        # \pi == pi via SymPy
        assert answers_equivalent("\\pi", "pi")


# ============================================================
#  Top-level reward
# ============================================================
class TestReward:
    def test_correct_gsm8k_no_box(self):
        # Correct answer with #### marker, no boxed, after format-cutoff step
        r = reward("Reasoning... #### 42", "42", step=200)
        assert r == 1.0

    def test_correct_boxed(self):
        r = reward("Working it out, \\boxed{42}", "42", step=200)
        assert r == 1.0

    def test_incorrect_no_box_no_format_bonus(self):
        r = reward("Reasoning... #### 99", "42", step=200)
        assert r == 0.0

    def test_incorrect_with_box_gets_format_bonus_early(self):
        r = reward("Wrong answer \\boxed{99}", "42", step=10)
        assert r == pytest.approx(0.1)

    def test_format_bonus_disabled_after_cutoff(self):
        r = reward("Wrong \\boxed{99}", "42", step=200)
        assert r == 0.0

    def test_format_bonus_at_exact_cutoff(self):
        # cutoff is exclusive: step < 100 gets bonus, step >= 100 does not
        r = reward("\\boxed{99}", "42", step=100)
        assert r == 0.0

    def test_correct_with_box_and_format_bonus(self):
        r = reward("\\boxed{42}", "42", step=10)
        assert r == pytest.approx(1.1)

    def test_gold_in_box(self):
        # MATH dataset puts gold in \boxed{} too — should still match
        r = reward("Reasoning... \\boxed{\\frac{1}{2}}",
                   "\\boxed{\\frac{1}{2}}", step=200)
        assert r == 1.0

    def test_fraction_correct(self):
        r = reward("\\boxed{1/2}", "0.5", step=200)
        assert r == 1.0


# ============================================================
#  Reward breakdown (diagnostic output)
# ============================================================
class TestRewardBreakdown:
    def test_returns_dataclass(self):
        out = reward_breakdown("\\boxed{42}", "42", step=10)
        assert isinstance(out, RewardBreakdown)

    def test_correct_with_box(self):
        out = reward_breakdown("\\boxed{42}", "42", step=10)
        assert out.correct
        assert out.has_box
        assert out.format_bonus == pytest.approx(0.1)
        assert out.total == pytest.approx(1.1)
        assert out.extracted == "42"
        assert out.extraction_method == "boxed"

    def test_gsm8k_extraction(self):
        out = reward_breakdown("Step by step\n#### 42", "42", step=10)
        assert out.correct
        assert not out.has_box
        assert out.format_bonus == 0.0
        assert out.total == 1.0
        assert out.extracted == "42"
        assert out.extraction_method == "gsm8k"

    def test_no_extraction(self):
        out = reward_breakdown("garbage no answer", "42", step=10)
        assert not out.correct
        assert out.extracted is None
        assert out.extraction_method == "none"
        assert out.total == 0.0


# ============================================================
#  Batch
# ============================================================
class TestBatchReward:
    def test_basic(self):
        completions = ["\\boxed{42}", "\\boxed{99}", "#### 7"]
        golds = ["42", "42", "7"]
        rewards = batch_reward(completions, golds, step=200)
        assert rewards == [1.0, 0.0, 1.0]

    def test_with_format_bonus(self):
        completions = ["\\boxed{42}", "\\boxed{99}", "no box"]
        golds = ["42", "42", "42"]
        rewards = batch_reward(completions, golds, step=10)
        assert rewards[0] == pytest.approx(1.1)
        assert rewards[1] == pytest.approx(0.1)
        assert rewards[2] == 0.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            batch_reward(["a", "b"], ["1"], step=0)

    def test_empty_lists(self):
        assert batch_reward([], [], step=0) == []
