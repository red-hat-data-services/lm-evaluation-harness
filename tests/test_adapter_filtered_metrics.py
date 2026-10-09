"""Regression coverage for incubation PR #167 on stable."""
import pytest
from main import _extract_lmeval_metrics, _normalize_lmeval_metrics


def test_mmlu_prox_custom_filter_is_exported():
    results = {"results": {"mmlu_prox_af": {"exact_match,custom-extract": 0.4}}}
    assert _extract_lmeval_metrics(results, "mmlu_prox_af") == {"exact_match": 0.4}


def test_named_filter_group_fallback():
    results = {
        "results": {
            "biology": {"exact_match,custom-extract": 0.25},
            "chemistry": {"exact_match,custom-extract": 0.75},
        },
        "group_subtasks": {"mmlu_prox_af": ["biology", "chemistry"]},
    }
    assert _extract_lmeval_metrics(results, "mmlu_prox_af") == {"exact_match": 0.5}


@pytest.mark.parametrize("reverse", [False, True])
def test_none_filter_preferred_without_double_counting(reverse):
    pairs = [("acc,none", 0.2), ("acc,flexible-extract", 0.8)]
    metrics = dict(reversed(pairs) if reverse else pairs)
    assert _normalize_lmeval_metrics(metrics) == {"acc": 0.2}
    results = {"results": {"one": metrics}, "group_subtasks": {"group": ["one"]}}
    assert _extract_lmeval_metrics(results, "group") == {"acc": 0.2}


@pytest.mark.parametrize("invalid", [None, "N/A", "invalid", float("nan"), float("inf")])
def test_invalid_metrics_are_skipped(invalid):
    metrics = {"exact_match,none": invalid, "exact_match,custom-extract": 0.6,
               "exact_match_stderr,custom-extract": invalid, "alias": "Af"}
    assert _normalize_lmeval_metrics(metrics) == {"exact_match": 0.6}


def test_existing_unfiltered_and_direct_group_scores_preserved():
    results = {"results": {"group": {"acc,none": 0.3}, "one": {"acc,none": 0.9}},
               "group_subtasks": {"group": ["one"]}}
    assert _extract_lmeval_metrics(results, "group") == {"acc": 0.3}


def test_no_scores_is_empty():
    assert _extract_lmeval_metrics({}, "missing") == {}
