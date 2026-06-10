"""Verifiable rewards for math-RL.

Two answer formats are supported:
    1. GSM8K-style:  "#### 42" or "answer is 42"
    2. MATH-style:   "\\boxed{42}", "\\boxed{\\frac{1}{2}}", "\\boxed{(3, \\pi)}"

Two verification backends are tried in order:
    1. `math_verify` (Apache-2.0, regex + ANTLR4-to-SymPy + numeric tolerance).
       This is the standard 2026 verifier used by TRL, verl, Open-R1.
       Requires Python ≥ 3.10 — only available on Modal in this project.
    2. SymPy direct (built-in fallback). Handles fractions, decimals,
       integers, and simple algebra. Misses some LaTeX edge cases that
       math_verify catches via ANTLR4.

The reward function is binary correctness + a small format bonus during the
first 100 RL steps to break the chicken-and-egg of "must produce \\boxed{}
to receive any reward."
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# ---- Backend selection ----
try:
    from math_verify import parse as _mv_parse, verify as _mv_verify  # type: ignore
    _MATH_VERIFY_AVAILABLE = True
except ImportError:
    _MATH_VERIFY_AVAILABLE = False

try:
    import sympy
    _SYMPY_AVAILABLE = True
except ImportError:
    _SYMPY_AVAILABLE = False


# Regex for GSM8K's "#### 42" terminator and the "answer is 42" fallback.
# Allow thousands separators (real GSM8K answers like "#### 1,000"); the commas
# are stripped after matching so the full integer survives.
GSM8K_FINAL_RE = re.compile(r"####\s*(-?\d[\d,]*(?:\.\d+)?)")
GSM8K_FALLBACK_RE = re.compile(
    r"(?:answer|the\s+answer)(?:\s+is)?\s*[:=]?\s*(-?\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)


def extract_gsm8k_answer(text: str) -> str | None:
    """
    Extract the integer/decimal answer from a GSM8K-formatted string.

    Tries (in order):
        1. The official `#### N` terminator.
        2. A loose "the answer is N" pattern as a fallback.

    Returns the matched numeric token as a stripped string, or None if no
    match. Returns the *first* match for `####`, *last* match for fallback
    (later answer-is statements override earlier ones).
    """
    m = GSM8K_FINAL_RE.search(text)
    if m:
        return m.group(1).replace(",", "")
    matches = GSM8K_FALLBACK_RE.findall(text)
    if matches:
        return matches[-1].replace(",", "")
    return None


def extract_boxed_answer(text: str) -> str | None:
    """
    Extract the contents of the LAST `\\boxed{...}` in `text`, handling
    nested braces correctly.

    LaTeX often nests braces inside `\\boxed{}` (e.g., `\\boxed{\\frac{1}{2}}`).
    A naive regex would stop at the first `}`. This walks character-by-
    character to balance braces.

    Returns the inner string (with surrounding whitespace stripped), or
    None if no `\\boxed{...}` is found or braces are unbalanced.
    """
    last_match: str | None = None
    i = 0
    needle = "\\boxed{"
    while True:
        start = text.find(needle, i)
        if start == -1:
            return last_match
        cursor = start + len(needle)
        depth = 1
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    last_match = text[start + len(needle):cursor].strip()
                    break
            cursor += 1
        if depth != 0:
            # Unbalanced — give up on this occurrence
            return last_match
        i = cursor + 1


def _sympy_normalize(expr_str: str):
    """Best-effort SymPy parse with light LaTeX preprocessing.

    Handles: integer, decimal, simple fraction (\\frac{a}{b}, a/b), pi/e
    constants. Does not handle full LaTeX — falls back to None on failure.
    """
    if not _SYMPY_AVAILABLE or expr_str is None:
        return None
    s = expr_str.strip()
    if not s:
        return None
    # Strip outer $...$ and \( \) common in LaTeX
    s = s.strip("$").strip()
    if s.startswith("\\(") and s.endswith("\\)"):
        s = s[2:-2].strip()
    # Common LaTeX fractions: \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\d?frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", s)
    # Strip \left \right
    s = s.replace("\\left", "").replace("\\right", "")
    # \cdot, \times -> *
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    # \pi -> pi
    s = s.replace("\\pi", "pi")
    # Strip stray backslashes that would confuse sympy
    s = re.sub(r"\\(?=[a-zA-Z])", " ", s)
    # Remove commas in numbers like "1,234"
    s = re.sub(r"(?<=\d),(?=\d{3}\b)", "", s)
    try:
        return sympy.sympify(s, rational=True)
    except (sympy.SympifyError, SyntaxError, TypeError, ValueError):
        return None


def answers_equivalent(
    pred: str | None,
    gold: str | None,
    rel_tol: float = 1e-6,
) -> bool:
    """
    True iff `pred` and `gold` represent the same mathematical answer.

    Tries (in order):
        1. Exact string match after whitespace strip.
        2. Numeric comparison after stripping common LaTeX wrappers.
        3. `math_verify` if installed (most robust).
        4. SymPy direct (`pred - gold == 0` symbolically) as fallback.

    Returns False on any parse failure.
    """
    if pred is None or gold is None:
        return False
    p = pred.strip()
    g = gold.strip()
    if not p or not g:
        return False

    # 1. Exact string match
    if p == g:
        return True

    # 2. Numeric direct compare (handles "42" vs "42.0", strips trailing zeros)
    try:
        return abs(float(p) - float(g)) <= rel_tol * max(1.0, abs(float(g)))
    except ValueError:
        pass

    # 3. math_verify (Modal-only)
    if _MATH_VERIFY_AVAILABLE:
        try:
            return bool(_mv_verify(_mv_parse(g), _mv_parse(p)))
        except Exception:
            pass

    # 4. SymPy fallback
    if _SYMPY_AVAILABLE:
        p_sym = _sympy_normalize(p)
        g_sym = _sympy_normalize(g)
        if p_sym is not None and g_sym is not None:
            try:
                diff = sympy.simplify(p_sym - g_sym)
                if diff == 0:
                    return True
                # Allow tiny numerical residual from float decimals
                f = float(diff)
                return abs(f) <= rel_tol * max(1.0, abs(float(g_sym)))
            except (TypeError, ValueError, sympy.SympifyError):
                pass

    return False


@dataclass
class RewardBreakdown:
    """Per-completion reward decomposition for diagnostics."""

    total: float
    correct: bool
    has_box: bool
    format_bonus: float
    extracted: str | None
    extraction_method: str  # "boxed" | "gsm8k" | "none"


def reward(
    completion: str,
    gold: str,
    step: int = 0,
    format_bonus_cutoff_steps: int = 100,
    format_bonus_value: float = 0.1,
    rel_tol: float = 1e-6,
) -> float:
    """
    Binary correctness reward + early-training format bonus.

    Args:
        completion: model output text
        gold: ground-truth answer string (may be raw integer for GSM8K or
              LaTeX for MATH)
        step: current RL training step. Format bonus is given only when
              step < format_bonus_cutoff_steps to break the cold-start
              dependency on producing well-formed output.
        format_bonus_cutoff_steps: disable format bonus after this step.
        format_bonus_value: small reward for producing `\\boxed{}` early.
        rel_tol: numerical tolerance for float comparisons.

    Returns:
        scalar reward in [0, 1 + format_bonus_value]:
            - 1.0 if the answer is correct
            - +format_bonus_value during early training if `\\boxed{}`
              is present, regardless of correctness
            - 0.0 otherwise
    """
    return reward_breakdown(
        completion, gold, step, format_bonus_cutoff_steps,
        format_bonus_value, rel_tol,
    ).total


def reward_breakdown(
    completion: str,
    gold: str,
    step: int = 0,
    format_bonus_cutoff_steps: int = 100,
    format_bonus_value: float = 0.1,
    rel_tol: float = 1e-6,
) -> RewardBreakdown:
    """Same as `reward` but returns the full diagnostic breakdown."""
    has_box = "\\boxed{" in completion
    fmt = format_bonus_value if (has_box and step < format_bonus_cutoff_steps) else 0.0

    # Try \boxed{} first (MATH style), then GSM8K-style ####.
    extracted = extract_boxed_answer(completion)
    method = "boxed" if extracted is not None else "none"
    if extracted is None:
        extracted = extract_gsm8k_answer(completion)
        method = "gsm8k" if extracted is not None else "none"

    # Gold may itself be wrapped in \boxed{} for MATH dataset entries — strip.
    gold_clean = extract_boxed_answer(gold) or extract_gsm8k_answer(gold) or gold
    correct = answers_equivalent(extracted, gold_clean, rel_tol=rel_tol)
    correct_score = 1.0 if correct else 0.0

    return RewardBreakdown(
        total=correct_score + fmt,
        correct=correct,
        has_box=has_box,
        format_bonus=fmt,
        extracted=extracted,
        extraction_method=method,
    )


def batch_reward(
    completions: list[str],
    golds: list[str],
    step: int = 0,
    **kwargs,
) -> list[float]:
    """Vectorized version of `reward`. Same args propagated per-item."""
    if len(completions) != len(golds):
        raise ValueError(
            f"batch_reward: length mismatch — {len(completions)} completions vs "
            f"{len(golds)} golds"
        )
    return [reward(c, g, step=step, **kwargs) for c, g in zip(completions, golds)]
