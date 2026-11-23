"""Report partial tinyTruthfulQA execution without fabricating its estimate."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import main


def job(limit=1, benchmark="tinyTruthfulQA"):
    return SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        benchmark_id=benchmark,
        benchmark_index=0,
        num_examples=limit,
        model=SimpleNamespace(name="test/model", url="http://localhost:8000/v1"),
        parameters={},
        exports=None,
        model_dump=lambda: {},
    )


@pytest.fixture
def adapter_run(monkeypatch, tmp_path):
    import datasets

    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False
    )
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    monkeypatch.setattr(
        main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None)
    )
    monkeypatch.setattr(main, "read_model_auth_key", lambda key: None)
    standard = MagicMock(return_value=0.75)
    aggregations = {"acc": standard}
    task = SimpleNamespace(aggregation=lambda: aggregations)
    manager = MagicMock()
    manager.load_task_or_group.return_value = {"tinyTruthfulQA": task}
    monkeypatch.setattr(main, "TaskManager", lambda: manager)
    monkeypatch.setattr(
        main, "_resolve_lmeval_task", lambda benchmark, manager: benchmark
    )
    monkeypatch.setattr(main, "_build_dataset_info", lambda *args: None)
    adapter = main.LMEvalAdapter.__new__(main.LMEvalAdapter)
    adapter._settings = main.AdapterSettings(mode="k8s")
    return adapter, MagicMock(), manager, standard


def evaluated_output(task, count):
    score = task.aggregation()["acc"]([0.5] * count)
    return {
        "results": {"tinyTruthfulQA": {"acc,none": score}},
        "samples": {"tinyTruthfulQA": [{"resps": [[(-0.5, False)]]}] * count},
        "n-samples": {"tinyTruthfulQA": {"original": 100, "effective": count}},
        "configs": {"tinyTruthfulQA": {"metric_list": [{"metric": "acc"}]}},
    }


@pytest.mark.parametrize("count", [1, 2, 99, 101])
@pytest.mark.parametrize("requested", [1, 100, None])
def test_actual_outcome_count_controls_skipped_aggregation(
    monkeypatch, adapter_run, count, requested
):
    adapter, callbacks, manager, standard = adapter_run
    evaluate = MagicMock(
        side_effect=lambda **kwargs: evaluated_output(kwargs["tasks"][0], count)
    )
    monkeypatch.setattr(main, "simple_evaluate", evaluate)
    result = adapter.run_benchmark_job(job(limit=requested), callbacks)

    assert evaluate.call_args.kwargs["limit"] == requested
    assert "predict_only" not in evaluate.call_args.kwargs
    standard.assert_not_called()
    assert result.results == []
    assert result.metrics_schema == []
    assert result.overall_score is None
    assert result.num_examples_evaluated == count
    info = result.additional_info
    assert info["num_examples_evaluated"] == count
    assert info["aggregation"]["status"] == "skipped"
    assert info["aggregation"]["required_examples"] == 100
    assert "exactly 100 predefined anchor outcomes" in info["aggregation"]["reason"]
    assert "evaluation_mode" not in info
    assert adapter.generate_additional_info(result) == info


@pytest.mark.parametrize("requested", [1, 100, 101, None])
def test_exactly_100_actual_outcomes_keep_standard_estimate(
    monkeypatch, adapter_run, requested
):
    adapter, callbacks, manager, standard = adapter_run
    monkeypatch.setattr(
        main,
        "simple_evaluate",
        lambda **kwargs: evaluated_output(kwargs["tasks"][0], 100),
    )
    result = adapter.run_benchmark_job(job(limit=requested), callbacks)
    standard.assert_called_once()
    assert len(standard.call_args.args[0]) == 100
    assert result.results[0].metric_name == "acc"
    assert result.results[0].metric_value == 0.75
    assert result.overall_score == 0.75
    assert result.num_examples_evaluated == 100
    assert result.additional_info is None
    assert "aggregation" not in adapter.generate_additional_info(result)


def test_other_benchmarks_keep_normal_evaluation(monkeypatch, adapter_run):
    adapter, callbacks, manager, standard = adapter_run
    evaluate = MagicMock(
        return_value={
            "results": {"arc_easy": {"acc,none": 0.75}},
            "samples": {"arc_easy": [{}]},
        }
    )
    monkeypatch.setattr(main, "simple_evaluate", evaluate)
    result = adapter.run_benchmark_job(job(benchmark="arc_easy"), callbacks)
    manager.load_task_or_group.assert_not_called()
    assert evaluate.call_args.kwargs["tasks"] == ["arc_easy"]
    assert result.overall_score == 0.75
    assert result.additional_info is None


def test_zero_outcomes_are_not_successful_execution(monkeypatch, adapter_run):
    adapter, callbacks, manager, standard = adapter_run
    monkeypatch.setattr(
        main,
        "simple_evaluate",
        lambda **kwargs: evaluated_output(kwargs["tasks"][0], 0),
    )
    with pytest.raises(RuntimeError, match="no completed sample outcomes"):
        adapter.run_benchmark_job(job(), callbacks)
    standard.assert_not_called()
    assert callbacks.report_status.call_args.args[0].status == main.JobStatus.FAILED


@pytest.mark.parametrize("count", [1, 100])
def test_failures_are_not_converted_to_skipped_aggregation(
    monkeypatch, adapter_run, count
):
    adapter, callbacks, manager, standard = adapter_run
    if count == 1:
        evaluate = MagicMock(side_effect=RuntimeError("model request failed"))
    else:
        standard.side_effect = RuntimeError("invalid calibration")

        def evaluate(**kwargs):
            return evaluated_output(kwargs["tasks"][0], count)

    monkeypatch.setattr(main, "simple_evaluate", evaluate)
    with pytest.raises(RuntimeError, match="model request failed|invalid calibration"):
        adapter.run_benchmark_job(job(limit=count), callbacks)
    assert callbacks.report_status.call_args.args[0].status == main.JobStatus.FAILED


def test_sdk_reports_completed_result_without_an_official_metric(
    monkeypatch, adapter_run
):
    from evalhub.models import PrimaryScore

    adapter, callbacks, manager, standard = adapter_run
    monkeypatch.setattr(
        main,
        "simple_evaluate",
        lambda **kwargs: evaluated_output(kwargs["tasks"][0], 1),
    )
    result = adapter.run_benchmark_job(job(), callbacks)
    monkeypatch.setattr(
        "evalhub.adapter.callbacks.EnvironmentCardMetadata.capture", lambda: None
    )
    reporter = main.DefaultCallbacks(
        job_id=result.id,
        benchmark_id=result.benchmark_id,
        provider_id="lm_evaluation_harness",
        sidecar_url="http://localhost:8080",
        auth_token="local-test-only",
        primary_score=PrimaryScore(metric="acc", lower_is_better=False),
    )
    reporter._http_client = MagicMock()
    reporter.report_results(result)
    event = reporter._http_client.post.call_args.kwargs["json"][
        "benchmark_status_event"
    ]
    assert event["status"] == "completed"
    assert event["id"] == "tinyTruthfulQA"
    assert event["provider_id"] == "lm_evaluation_harness"
    assert event["metrics"] == {}
    assert "overall_score" not in event
    assert not event.get("metrics_schema")
    assert event["additional_info"]["aggregation"]["status"] == "skipped"
    assert event["additional_info"]["num_examples_evaluated"] == 1


@pytest.mark.parametrize(
    "limit,population",
    [(1, 100), (99, 100), (100, 100), (100, 99), (None, 100), (None, 99), (101, 101)],
)
def test_real_harness_counts_outcomes_after_inference(
    monkeypatch, adapter_run, limit, population
):
    """Use real task/request/evaluation code with local model/estimator doubles."""
    from pathlib import Path

    from datasets import Dataset, DatasetDict

    from lm_eval import evaluator, utils
    from lm_eval.api.task import ConfigurableTask
    from lm_eval.models.dummy import DummyLM

    adapter, callbacks, manager, standard = adapter_run
    data = DatasetDict(
        validation=Dataset.from_list(
            [
                {
                    "question": f"Local protocol fixture {i}?",
                    "mc1_targets": {"choices": ["Yes", "No"], "labels": [1, 0]},
                    "mc2_targets": {"choices": ["Yes", "No"], "labels": [1, 0]},
                }
                for i in range(population)
            ]
        )
    )

    def local_download(self, *args, **kwargs):
        self.dataset = data

    monkeypatch.setattr(ConfigurableTask, "download", local_download)
    task_path = (
        Path(__file__).parents[1]
        / "lm_eval/tasks/tinyBenchmarks/tinyTruthfulQA_mc2.yaml"
    )
    task_config = utils.load_yaml_config(str(task_path))
    aggregated_counts = []

    def recording_aggregation(items):
        aggregated_counts.append(len(items))
        assert len(items) == 100
        return 0.5

    task_config["metric_list"][0]["aggregation"] = recording_aggregation
    task = ConfigurableTask(config=task_config)
    manager.load_task_or_group.return_value = {"tinyTruthfulQA": task}

    class RecordingLM(DummyLM):
        request_count = 0

        def loglikelihood(self, requests, **kwargs):
            self.request_count += len(requests)
            return [(-0.5, False) for _ in requests]

    model = RecordingLM()

    def run_actual_harness(**kwargs):
        kwargs.update(model=model, task_manager=None, bootstrap_iters=0)
        return evaluator.simple_evaluate(**kwargs)

    monkeypatch.setattr(main, "simple_evaluate", run_actual_harness)
    result = adapter.run_benchmark_job(job(limit=limit), callbacks)
    actual = min(limit, population) if limit is not None else population
    assert result.num_examples_evaluated == actual
    assert model.request_count == actual * 2
    if actual != 100:
        assert aggregated_counts == []
        assert result.results == []
        assert result.overall_score is None
        assert result.additional_info["aggregation"]["status"] == "skipped"
    else:
        assert aggregated_counts == [100]
        assert result.results[0].metric_name == "acc"
        assert result.overall_score == 0.5
