"""Offline inputs must remain unchanged when tasks map or filter staged data."""

import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import datasets
import pytest

from lm_eval.api.task import ConfigurableTask, _load_datasetdict_from_offline_bundle


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


def _task_for_splits(dataset, process_docs, **split_names):
    # Exercise document access without model/task-registry initialization.
    task = ConfigurableTask.__new__(ConfigurableTask)
    task._config = SimpleNamespace(
        process_docs=process_docs,
        training_split=split_names.get("training", "train"),
        validation_split=split_names.get("validation", "train"),
        test_split=split_names.get("test"),
        fewshot_split=split_names.get("fewshot", "train"),
    )
    task.dataset = dataset
    task._processed_splits = {}
    return task


@pytest.mark.parametrize("variant", ["cot_zeroshot", "zeroshot"])
def test_gpqa_documents_stay_stable_without_disk_caching(tmp_path, variant):
    utils_path = Path(__file__).parents[1] / "lm_eval" / "tasks" / "gpqa" / variant / "utils.py"
    spec = importlib.util.spec_from_file_location(f"gpqa_{variant}_utils", utils_path)
    gpqa = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpqa)
    bundle = tmp_path / "gpqa"
    datasets.DatasetDict({"train": datasets.Dataset.from_dict({
        "Incorrect Answer 1": ["wrong1"] * 10,
        "Incorrect Answer 2": ["wrong2"] * 10,
        "Incorrect Answer 3": ["wrong3"] * 10,
        "Correct Answer": ["correct"] * 10,
    })}).save_to_disk(str(bundle))
    calls = []

    def process_docs(docs):
        calls.append(True)
        return gpqa.process_docs(docs)

    previously_enabled = datasets.is_caching_enabled()
    random_state = random.getstate()
    random.seed(42)
    try:
        task = _task_for_splits(_load_datasetdict_from_offline_bundle(bundle), process_docs)
        request_docs = task.eval_docs
        expected = request_docs.to_dict()
        # Simulate unrelated random draws between request building and scoring.
        for _ in range(100):
            random.random()
        scoring_docs = task.eval_docs
        assert scoring_docs is request_docs
        assert scoring_docs.to_dict() == expected
        assert task.training_docs() is request_docs
        assert task.fewshot_docs() is request_docs
        assert len(calls) == 1
        for row in scoring_docs:
            correct_letter = ord(row["answer"][1]) - ord("A") + 1
            assert row[f"choice{correct_letter}"] == "correct"
    finally:
        random.setstate(random_state)
        if previously_enabled:
            datasets.enable_caching()


@pytest.mark.parametrize("process", [True, False])
def test_processed_splits_are_separate_and_task_local(process):
    source = datasets.DatasetDict({
        "train": datasets.Dataset.from_dict({"value": [1]}),
        "test": datasets.Dataset.from_dict({"value": [2]}),
    })
    calls = []

    def process_docs(docs):
        calls.append(docs[0]["value"])
        return datasets.Dataset.from_dict({"value": [docs[0]["value"] + 10]})

    processor = process_docs if process else None
    task = _task_for_splits(source, processor, test="test")
    train = task.training_docs()
    test = task.test_docs()
    assert task.training_docs() is train
    assert task.validation_docs() is train
    assert task.fewshot_docs() is train
    assert task.test_docs() is test
    assert train[0]["value"] == (11 if process else 1)
    assert test[0]["value"] == (12 if process else 2)
    other = _task_for_splits(source, processor, test="test")
    other.test_docs()
    assert calls == ([1, 2, 2] if process else [])
