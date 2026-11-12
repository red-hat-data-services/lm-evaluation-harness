"""MLflow export failures must not discard completed evaluation results."""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
from evalhub.adapter import DefaultCallbacks
from evalhub.adapter import mlflow as sdk_mlflow

import main


@pytest.fixture
def export_setup(monkeypatch):
    results = SimpleNamespace(
        benchmark_id="benchmark",
        model_name="model",
        num_examples_evaluated=1,
        duration_seconds=1.0,
        results=[],
        overall_score=0.75,
        eval_card=None,
        env_card=None,
        mlflow_run_id=None,
    )
    adapter = Mock()
    adapter.job_spec.parameters = {}
    adapter.job_spec.tags = []
    adapter.job_spec.experiment_name = "experiment"
    adapter.run_benchmark_job.return_value = results
    adapter.mlflow_artifacts = [
        sdk_mlflow.MlflowArtifact(path="results.json", content=b"{}")
    ]
    callbacks = DefaultCallbacks(job_id="job", benchmark_id="benchmark")
    monkeypatch.setattr(callbacks, "report_status", Mock())
    monkeypatch.setattr(callbacks, "report_results", Mock())
    monkeypatch.setattr(main, "LMEvalAdapter", Mock(return_value=adapter))
    monkeypatch.setattr(
        main.DefaultCallbacks, "from_adapter", Mock(return_value=callbacks)
    )
    client = MagicMock()
    client.__enter__.return_value = client
    client.start_run.return_value.__enter__.return_value = "run-1"
    monkeypatch.setattr(sdk_mlflow, "MlflowClient", Mock(return_value=client))
    return adapter, callbacks, client, results


@pytest.mark.parametrize(
    "operation", ["get_or_create_experiment", "log_batch", "upload_artifact"]
)
def test_mlflow_failure_still_reports_results(export_setup, caplog, operation):
    _, callbacks, client, results = export_setup
    getattr(client, operation).side_effect = RuntimeError("token=SECRET")

    assert main.main() == 0

    getattr(client, operation).assert_called_once()
    callbacks.report_results.assert_called_once_with(results)
    callbacks.report_status.assert_not_called()
    assert callbacks.mlflow._callbacks is callbacks
    assert results.mlflow_run_id is None
    if operation == "upload_artifact":
        client.log_batch.assert_called_once()
    warnings = [r.message for r in caplog.records if r.name == main.logger.name]
    assert any("continuing with evaluation results" in msg for msg in warnings)
    assert all("SECRET" not in msg for msg in warnings)


def test_success_retains_run_id_and_restores_callback(export_setup):
    _, callbacks, client, results = export_setup

    assert main.main() == 0

    assert results.mlflow_run_id == "run-1"
    callbacks.report_results.assert_called_once_with(results)
    callbacks.report_status.assert_not_called()
    assert callbacks.mlflow._callbacks is callbacks
    client.log_batch.assert_called_once()
    client.upload_artifact.assert_called_once()


def test_no_experiment_skips_mlflow(export_setup):
    adapter, callbacks, client, results = export_setup
    adapter.job_spec.experiment_name = None

    assert main.main() == 0

    client.__enter__.assert_not_called()
    callbacks.report_results.assert_called_once_with(results)
    assert callbacks.mlflow._callbacks is callbacks


@pytest.mark.parametrize("failure", ["evaluation", "report_results"])
def test_non_mlflow_failures_remain_fatal(export_setup, failure):
    adapter, callbacks, client, _ = export_setup
    if failure == "evaluation":
        adapter.run_benchmark_job.side_effect = RuntimeError("evaluation failed")
    else:
        callbacks.report_results.side_effect = RuntimeError("delivery failed")

    assert main.main() == 1

    if failure == "evaluation":
        client.__enter__.assert_not_called()
        callbacks.report_results.assert_not_called()
    assert callbacks.mlflow._callbacks is callbacks
