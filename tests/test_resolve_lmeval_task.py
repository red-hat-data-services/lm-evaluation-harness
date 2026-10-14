"""Tests for eval-hub benchmark_id -> lm-eval task resolution."""

import pytest

from lm_eval.tasks import TaskManager
from main import _resolve_lmeval_task


@pytest.fixture(scope="module")
def task_manager() -> TaskManager:
    return TaskManager()


def test_winogender_resolves_to_winogender_all(task_manager: TaskManager) -> None:
    assert _resolve_lmeval_task("winogender", task_manager) == "winogender_all"


def test_arc_easy_passthrough(task_manager: TaskManager) -> None:
    assert _resolve_lmeval_task("arc_easy", task_manager) == "arc_easy"


def test_longbench_catalog_alias(task_manager: TaskManager) -> None:
    assert _resolve_lmeval_task("longbenchv2", task_manager) == "longbench2"
    assert "longbench2" in task_manager.all_tasks


def test_longbench_harness_id_passthrough(task_manager: TaskManager) -> None:
    assert _resolve_lmeval_task("longbench2", task_manager) == "longbench2"


def test_unmapped_tag_raises(task_manager: TaskManager) -> None:
    # "social_bias" is a tag in lm-eval but not in eval-hub catalog aliases.
    with pytest.raises(ValueError, match="lm-eval tag"):
        _resolve_lmeval_task("social_bias", task_manager)
