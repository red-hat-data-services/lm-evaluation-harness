"""Tests for job spec path validation in ``_read_job_spec_parameters_from_path``."""

from pathlib import Path

import pytest

import main


def test_resolve_rejects_path_outside_meta() -> None:
    assert main._resolve_job_spec_path_for_read("/tmp/job.json") is None


def test_resolve_rejects_dotdot_escape() -> None:
    assert main._resolve_job_spec_path_for_read("/meta/../etc/passwd") is None


def test_resolve_accepts_meta_job_json() -> None:
    resolved = main._resolve_job_spec_path_for_read("/meta/job.json")
    assert resolved is not None
    assert resolved == Path("/meta/job.json").resolve()


def test_read_parameters_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(main, "_JOB_SPEC_ALLOWED_ROOT", tmp_path)
    job = tmp_path / "job.json"
    job.write_text('{"parameters": {"limit": 3}}', encoding="utf-8")
    assert main._read_job_spec_parameters_from_path(str(job)) == {"limit": 3}


def test_read_parameters_missing_file_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(main, "_JOB_SPEC_ALLOWED_ROOT", tmp_path)
    assert main._read_job_spec_parameters_from_path(str(tmp_path / "nope.json")) == {}


def test_read_parameters_invalid_json_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(main, "_JOB_SPEC_ALLOWED_ROOT", tmp_path)
    job = tmp_path / "job.json"
    job.write_text("{not-json", encoding="utf-8")
    assert main._read_job_spec_parameters_from_path(str(job)) == {}
    err = capsys.readouterr().err
    assert "invalid JSON" in err
