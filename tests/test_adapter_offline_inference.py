"""Tests for HF offline auto-detection from ``parameters.tokenizer`` + ``/test_data`` (no ``offline`` flag)."""

from pathlib import Path

import pytest

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
