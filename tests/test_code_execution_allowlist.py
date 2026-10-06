"""Tests for the EvalHub adapter code-execution allow list."""

import os

import pytest

from main import _code_eval_environment, _needs_code_execution


@pytest.mark.parametrize(
    "benchmark_id",
    ["humaneval", "humaneval_instruct", "mbpp"],
)
def test_code_execution_benchmarks_are_allow_listed(benchmark_id: str) -> None:
    assert _needs_code_execution(benchmark_id)


@pytest.mark.parametrize(
    "benchmark_id",
    ["arc_easy", "ethics_cm", "mbpp_plus", "unknown"],
)
def test_other_benchmarks_are_not_allow_listed(benchmark_id: str) -> None:
    assert not _needs_code_execution(benchmark_id)


@pytest.mark.parametrize(
    ("benchmark_id", "expected"),
    [("humaneval", "1"), ("mbpp", "1"), ("arc_easy", "0")],
)
def test_code_eval_environment_is_scoped_and_restored(
    monkeypatch: pytest.MonkeyPatch,
    benchmark_id: str,
    expected: str,
) -> None:
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "previous")

    with _code_eval_environment(benchmark_id):
        assert os.environ["HF_ALLOW_CODE_EVAL"] == expected

    assert os.environ["HF_ALLOW_CODE_EVAL"] == "previous"


def test_code_eval_environment_restores_an_unset_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_ALLOW_CODE_EVAL", raising=False)

    with _code_eval_environment("humaneval"):
        assert os.environ["HF_ALLOW_CODE_EVAL"] == "1"

    assert "HF_ALLOW_CODE_EVAL" not in os.environ


def test_code_eval_environment_restores_after_an_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_ALLOW_CODE_EVAL", "previous")

    with pytest.raises(RuntimeError, match="boom"):
        with _code_eval_environment("humaneval"):
            raise RuntimeError("boom")

    assert os.environ["HF_ALLOW_CODE_EVAL"] == "previous"
