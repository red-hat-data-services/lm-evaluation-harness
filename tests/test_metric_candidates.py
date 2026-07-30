"""Unit tests for skipping non-finite lm-eval metric values (BBQ NaN/Inf)."""

import pytest

from main import _to_finite_float


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (0.9, 0.9),
        (1, 1.0),
        ("0.5", 0.5),
        ("N/A", None),
        (None, None),
        (float("nan"), None),
        (float("inf"), None),
        (float("-inf"), None),
        ("not-a-number", None),
    ],
)
def test_to_finite_float(raw: object, expected: float | None) -> None:
    got = _to_finite_float(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)
