"""Offline inputs must remain unchanged when tasks map or filter staged data."""

from pathlib import Path

import datasets
import pytest

from lm_eval.api.task import _load_datasetdict_from_offline_bundle


@pytest.mark.parametrize("operation", ["map", "filter"])
def test_staged_dataset_transform_uses_scratch(tmp_path: Path, operation: str):
    bundle = tmp_path / "test_data" / "example--dataset--All"
    datasets.DatasetDict(
        {"test": datasets.Dataset.from_dict({"value": [1, 2, 3]})}
    ).save_to_disk(str(bundle))
    original_files = {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()}
    for path in bundle.rglob("*"):
        path.chmod(0o555 if path.is_dir() else 0o444)
    bundle.chmod(0o555)
    previously_enabled = datasets.is_caching_enabled()
    try:
        loaded = _load_datasetdict_from_offline_bundle(bundle)["test"]
        source_files = {cache["filename"] for cache in loaded.cache_files}
        if operation == "map":
            transformed = loaded.map(lambda row: {"value": row["value"] + 1})
            assert transformed["value"] == [2, 3, 4]
        else:
            transformed = loaded.filter(lambda row: row["value"] > 1)
            assert transformed["value"] == [2, 3]
        # filter keeps the original Arrow data and writes only its row-index
        # cache. Reading source files is fine; newly created files must be scratch.
        processing_files = [cache for cache in transformed.cache_files if cache["filename"] not in source_files]
        assert processing_files
        for cache in processing_files:
            assert not Path(cache["filename"]).resolve().is_relative_to(bundle.resolve())
        assert {p.relative_to(bundle): p.read_bytes() for p in bundle.rglob("*") if p.is_file()} == original_files
    finally:
        if previously_enabled:
            datasets.enable_caching()
        for path in bundle.rglob("*"):
            path.chmod(0o755 if path.is_dir() else 0o644)
        bundle.chmod(0o755)
