"""Result files are saved in the appropriate directory for each runtime."""

import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import datasets
import pytest
from evalhub.adapter import OCIArtifactResult

import main


@pytest.mark.parametrize("mode", ["local", "k8s", None])
@pytest.mark.parametrize("oci_enabled", [False, True])
def test_benchmark_saves_results_in_runtime_output_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str | None, oci_enabled: bool
) -> None:
    job_dir = tmp_path / "evalhub-jobs/job-1/0/provider/benchmark"
    job_path = job_dir / "meta/job.json"
    job_path.parent.mkdir(parents=True)
    job_path.write_text(
        json.dumps(
            {
                "id": "job-1",
                "provider_id": "provider",
                "benchmark_id": "benchmark",
                "benchmark_index": 0,
                "model": {"name": "test-model", "url": "http://localhost:8080"},
                "parameters": {},
                "callback_url": "http://localhost:8080",
                "exports": {
                    "oci": {
                        "coordinates": {
                            "oci_host": "registry.example.com",
                            "oci_repository": "results",
                        }
                    }
                }
                if oci_enabled
                else None,
            }
        ),
        encoding="utf-8",
    )
    adapter_dir = tmp_path / "adapter"
    monkeypatch.setattr(main, "__file__", str(adapter_dir / "main.py"))
    if mode is None:
        monkeypatch.delenv("EVALHUB_MODE", raising=False)
    else:
        monkeypatch.setenv("EVALHUB_MODE", mode)
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_path))
    # Dataset loading is mocked; provide the configuration used by the adapter
    # even when the installed datasets version no longer exposes this flag.
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False
    )
    monkeypatch.setattr(
        main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None)
    )
    monkeypatch.setattr(main, "read_model_auth_key", lambda key: None)
    monkeypatch.setattr(main, "build_lmeval_config", lambda config: ("hf", {}, {}))
    monkeypatch.setattr(main, "TaskManager", Mock())
    monkeypatch.setattr(
        main, "_resolve_lmeval_task", lambda benchmark, manager: benchmark
    )
    monkeypatch.setattr(main, "_build_additional_info", lambda **kwargs: {})
    evaluate = Mock(
        return_value={
            "results": {"benchmark": {"acc,none": 0.75}},
            "samples": {"benchmark": [{"doc_id": 0}]},
        }
    )
    monkeypatch.setattr(main, "simple_evaluate", evaluate)
    callbacks = Mock()
    callbacks.create_oci_artifact.return_value = OCIArtifactResult(
        digest="sha256:test", reference="registry.example.com/results@sha256:test"
    )
    settings = main._load_adapter_settings()
    adapter = main.LMEvalAdapter(settings=settings)
    assert adapter.settings is settings
    assert adapter.settings.mode == (mode or "k8s")

    expected_dir = (job_dir if mode == "local" else adapter_dir) / "output"
    nested_dir = expected_dir / "details"
    nested_dir.mkdir(parents=True)
    (nested_dir / "samples.jsonl").write_bytes(b'{"doc_id": 0}\n')
    (expected_dir / "linked.jsonl").symlink_to(nested_dir / "samples.jsonl")

    results = adapter.run_benchmark_job(adapter.job_spec, callbacks)

    other_dir = (adapter_dir if mode == "local" else job_dir) / "output"
    saved = json.loads(
        (expected_dir / "results_job-1.json").read_text(encoding="utf-8")
    )
    assert saved["id"] == results.id == "job-1"
    assert saved["overall_score"] == results.overall_score == 0.75
    assert not other_dir.exists()
    evaluate.assert_called_once()
    artifacts = {artifact.path: artifact for artifact in adapter.mlflow_artifacts}
    assert set(artifacts) == {"results_job-1.json", "details/samples.jsonl"}
    for path, artifact in artifacts.items():
        assert artifact.content == (expected_dir / path).read_bytes()
    assert artifacts["results_job-1.json"].content_type == "application/json"
    if oci_enabled:
        callbacks.create_oci_artifact.assert_called_once()
        assert (
            callbacks.create_oci_artifact.call_args.args[0].files_path == expected_dir
        )
    else:
        callbacks.create_oci_artifact.assert_not_called()


def test_main_passes_output_artifacts_to_mlflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = Mock()
    adapter.job_spec.parameters = {}
    results = adapter.run_benchmark_job.return_value
    results.duration_seconds = 1.0
    callbacks = Mock()
    callbacks.mlflow.save.return_value = "run-1"
    monkeypatch.setattr(main, "LMEvalAdapter", Mock(return_value=adapter))
    monkeypatch.setattr(
        main.DefaultCallbacks, "from_adapter", Mock(return_value=callbacks)
    )

    assert main.main() == 0

    callbacks.mlflow.save.assert_called_once_with(
        results, adapter.job_spec, artifacts=adapter.mlflow_artifacts
    )
    assert results.mlflow_run_id == "run-1"
    callbacks.report_results.assert_called_once_with(results)


@pytest.mark.parametrize("replacement", ["symlink", "fifo"])
def test_output_read_rejects_path_replacement(tmp_path: Path, replacement: str) -> None:
    output_dir = tmp_path / "output"
    nested = output_dir / "nested"
    nested.mkdir(parents=True)
    file = nested / "sample.json"
    file.write_bytes(b"original")
    # Simulate replacement after the caller's file checks.
    assert file.is_file() and not file.is_symlink()
    file.unlink()
    external = tmp_path / "external"
    external.mkdir()
    (external / file.name).write_bytes(b"secret")
    if replacement == "symlink":
        file.symlink_to(external / file.name)
    else:
        os.mkfifo(file)

    with pytest.raises((OSError, ValueError)):
        main._read_output_file(file)
