"""Tests for HF offline auto-detection from staged tokenizer assets under ``/test_data``."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main
from main import _infer_auto_offline_from_local_test_data


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("{}", encoding="utf-8")


@pytest.fixture
def fake_test_data(tmp_path: Path) -> Path:
    """Typical EvalHub sync: ``tokenizer/`` + slug bundle with ``dataset_dict.json``."""
    root = tmp_path / "test_data"
    tok = root / "tokenizer"
    tok.mkdir(parents=True)
    _touch(tok / "config.json")
    bundle = root / "allenai--ai2_arc--ARC-Easy"
    bundle.mkdir(parents=True)
    _touch(bundle / "dataset_dict.json")
    return root


def test_infer_true_when_tokenizer_and_slug_bundle(fake_test_data: Path) -> None:
    tok_path = fake_test_data / "tokenizer"
    params = {"tokenizer": str(tok_path.resolve())}
    assert _infer_auto_offline_from_local_test_data(params, test_data_root=fake_test_data)


def test_nested_parameters_tokenizer_does_not_trigger_offline(fake_test_data: Path) -> None:
    """Nested ``parameters.parameters.tokenizer`` is not used by ``build_lmeval_config``; infer must match."""
    tok_path = fake_test_data / "tokenizer"
    params = {
        "tokenizer": "google/flan-t5-small",
        "parameters": {"tokenizer": str(tok_path.resolve())},
    }
    assert not _infer_auto_offline_from_local_test_data(
        params, test_data_root=fake_test_data
    )


def test_infer_false_when_hf_model_id(fake_test_data: Path) -> None:
    params = {"tokenizer": "google/flan-t5-small"}
    assert not _infer_auto_offline_from_local_test_data(
        params, test_data_root=fake_test_data
    )


def test_infer_false_when_tokenizer_path_missing(fake_test_data: Path) -> None:
    missing = fake_test_data / "nope"
    params = {"tokenizer": str(missing.resolve())}
    assert not _infer_auto_offline_from_local_test_data(
        params, test_data_root=fake_test_data
    )


def test_infer_false_when_no_dataset_dict_sibling(fake_test_data: Path) -> None:
    """No ``dataset_dict.json`` under a sibling dir → do not infer offline."""
    bundle = fake_test_data / "allenai--ai2_arc--ARC-Easy"
    (bundle / "dataset_dict.json").unlink(missing_ok=True)
    bundle.rmdir()
    tok_path = fake_test_data / "tokenizer"
    params = {"tokenizer": str(tok_path.resolve())}
    assert not _infer_auto_offline_from_local_test_data(
        params, test_data_root=fake_test_data
    )


def test_infer_false_when_tokenizer_is_root_only(fake_test_data: Path) -> None:
    params = {"tokenizer": str(fake_test_data.resolve())}
    assert not _infer_auto_offline_from_local_test_data(
        params, test_data_root=fake_test_data
    )


def test_infer_false_when_path_outside_root(tmp_path: Path) -> None:
    root = tmp_path / "test_data"
    root.mkdir()
    bundle = root / "some--dataset"
    bundle.mkdir(parents=True)
    _touch(bundle / "dataset_dict.json")
    outside = tmp_path / "other" / "tok"
    outside.mkdir(parents=True)
    _touch(outside / "config.json")
    params = {"tokenizer": str(outside.resolve())}
    assert not _infer_auto_offline_from_local_test_data(
        params, test_data_root=root
    )


@pytest.mark.parametrize("params", [{}, {"tokenizer": ""}])
def test_toxigen_uses_staged_default_tokenizer_without_explicit_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, params: dict[str, str]
) -> None:
    """Offline ToxiGen uses staged dataset and tokenizer assets."""
    root = tmp_path / "test_data"
    tokenizer = root / "tokenizer"
    _touch(tokenizer / "config.json")
    _touch(root / "skg--toxigen-data--annotated" / "dataset_dict.json")
    monkeypatch.setattr(main, "_TEST_DATA_DIR", str(root))

    assert _infer_auto_offline_from_local_test_data(params, test_data_root=root)

    model = SimpleNamespace(
        name="meta-llama/Llama-3.1-8B-Instruct",
        url="https://target.example/v1",
    )
    job = SimpleNamespace(model=model, parameters=params, benchmark_id="toxigen")
    _, model_args, _ = main.build_lmeval_config(job)
    assert model_args["tokenizer"] == str(tokenizer.resolve())


def test_staged_default_tokenizer_does_not_enable_offline_without_dataset(
    tmp_path: Path,
) -> None:
    root = tmp_path / "test_data"
    _touch(root / "tokenizer" / "config.json")

    assert not _infer_auto_offline_from_local_test_data({}, test_data_root=root)


def test_explicit_tokenizer_model_id_takes_precedence_over_staged_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "test_data"
    _touch(root / "tokenizer" / "config.json")
    _touch(root / "skg--toxigen-data--annotated" / "dataset_dict.json")
    monkeypatch.setattr(main, "_TEST_DATA_DIR", str(root))

    model = SimpleNamespace(name="target-model", url="https://target.example/v1")
    job = SimpleNamespace(
        model=model,
        parameters={"tokenizer": "custom/tokenizer"},
        benchmark_id="toxigen",
    )
    _, model_args, _ = main.build_lmeval_config(job)
    assert model_args["tokenizer"] == "custom/tokenizer"
    assert not _infer_auto_offline_from_local_test_data(
        job.parameters, test_data_root=root
    )


def test_seed_does_not_enable_offline_when_job_spec_cannot_be_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "test_data"
    _touch(root / "tokenizer" / "config.json")
    _touch(root / "skg--toxigen-data--annotated" / "dataset_dict.json")
    monkeypatch.setattr(main, "_TEST_DATA_DIR", str(root))
    monkeypatch.setattr(main, "_JOB_SPEC_ALLOWED_ROOT", tmp_path)
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(tmp_path / "missing-job.json"))
    offline_vars = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_EVALUATE_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    original_env = {name: main.os.environ.get(name) for name in offline_vars}
    with monkeypatch.context() as env_patch:
        for name in offline_vars:
            env_patch.setenv(name, "")
            env_patch.delenv(name)

        main._seed_hf_offline_before_lm_eval_import()

        assert all(name not in main.os.environ for name in offline_vars)

    assert {name: main.os.environ.get(name) for name in offline_vars} == original_env


def test_seed_offline_environment_for_staged_toxigen_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "test_data"
    _touch(root / "tokenizer" / "config.json")
    _touch(root / "skg--toxigen-data--annotated" / "dataset_dict.json")
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps({"parameters": {}}), encoding="utf-8")
    monkeypatch.setattr(main, "_TEST_DATA_DIR", str(root))
    monkeypatch.setattr(main, "_JOB_SPEC_ALLOWED_ROOT", tmp_path)
    monkeypatch.setenv("EVALHUB_JOB_SPEC_PATH", str(job_path))
    offline_vars = (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HF_DATASETS_CACHE",
        "HF_HUB_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "HF_EVALUATE_OFFLINE",
        "TRANSFORMERS_OFFLINE",
    )
    original_env = {name: main.os.environ.get(name) for name in offline_vars}
    with monkeypatch.context() as env_patch:
        for name in offline_vars:
            env_patch.setenv(name, "")
            env_patch.delenv(name)

        main._seed_hf_offline_before_lm_eval_import()

        assert main.os.environ["HF_HOME"] == str(root)
        assert main.os.environ["HF_HUB_CACHE"] == str(root / "hub")
        assert main.os.environ["HF_DATASETS_CACHE"] == str(root / "datasets")
        assert main.os.environ["HF_HUB_OFFLINE"] == "1"
        assert main.os.environ["HF_DATASETS_OFFLINE"] == "1"
        assert main.os.environ["HF_EVALUATE_OFFLINE"] == "1"
        assert main.os.environ["TRANSFORMERS_OFFLINE"] == "1"

    assert {name: main.os.environ.get(name) for name in offline_vars} == original_env
