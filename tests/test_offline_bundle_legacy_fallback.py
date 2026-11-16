"""Offline bundle resolution must prefer the namespaced slug and fall back to the legacy, un-namespaced slug."""

from pathlib import Path

import datasets

from lm_eval.api.task import _legacy_fallback_allowed, _resolve_offline_bundle_dir


def _stage_bundle(root: Path, slug: str) -> Path:
    bundle = root / slug
    datasets.DatasetDict(
        {"test": datasets.Dataset.from_dict({"value": [1]})}
    ).save_to_disk(str(bundle))
    return bundle


def test_prefers_namespaced_bundle_when_both_exist(tmp_path: Path):
    namespaced = _stage_bundle(tmp_path, "Rowan--hellaswag")
    _stage_bundle(tmp_path, "hellaswag")

    resolved = _resolve_offline_bundle_dir(str(tmp_path), "Rowan/hellaswag", None)

    assert resolved == namespaced


def test_falls_back_to_legacy_bundle_when_namespaced_missing(tmp_path: Path):
    legacy = _stage_bundle(tmp_path, "hellaswag")

    resolved = _resolve_offline_bundle_dir(str(tmp_path), "Rowan/hellaswag", None)

    assert resolved == legacy


def test_returns_none_when_neither_bundle_exists(tmp_path: Path):
    assert _resolve_offline_bundle_dir(str(tmp_path), "Rowan/hellaswag", None) is None


def test_no_legacy_fallback_for_already_unnamespaced_path(tmp_path: Path):
    # dataset_path without a "/" has no legacy form to fall back to.
    assert _resolve_offline_bundle_dir(str(tmp_path), "hellaswag", None) is None


def test_no_legacy_fallback_for_unmapped_namespaced_path(tmp_path: Path):
    # A custom namespaced id must never resolve to an unrelated legacy bundle.
    _stage_bundle(tmp_path, "hellaswag")

    assert _resolve_offline_bundle_dir(str(tmp_path), "other/hellaswag", None) is None


def test_legacy_fallback_disabled_when_only_legacy_bundle_exists(tmp_path: Path):
    _stage_bundle(tmp_path, "hellaswag")

    resolved = _resolve_offline_bundle_dir(
        str(tmp_path), "Rowan/hellaswag", None, allow_legacy_fallback=False
    )

    assert resolved is None


def test_namespaced_bundle_still_returned_when_legacy_fallback_disabled(
    tmp_path: Path,
):
    namespaced = _stage_bundle(tmp_path, "Rowan--hellaswag")
    _stage_bundle(tmp_path, "hellaswag")

    resolved = _resolve_offline_bundle_dir(
        str(tmp_path), "Rowan/hellaswag", None, allow_legacy_fallback=False
    )

    assert resolved == namespaced


def test_legacy_fallback_allowed_with_no_dataset_kwargs():
    assert _legacy_fallback_allowed(None) is True
    assert _legacy_fallback_allowed({}) is True
    assert _legacy_fallback_allowed({"revision": None}) is True


def test_legacy_fallback_disallowed_when_data_selection_kwargs_set():
    assert _legacy_fallback_allowed({"revision": "abc"}) is False
    assert _legacy_fallback_allowed({"data_files": "x.json"}) is False
    assert _legacy_fallback_allowed({"data_dir": "d"}) is False
