import json

import pytest

from lm_eval.decontamination.decontaminate import (
    _load_lookup_cache,
    _load_overlap_cache,
    get_train_overlap,
)


def test_get_train_overlap_writes_and_reads_json_caches(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    ngrams_path = tmp_path / "ngrams"
    ngrams_path.mkdir()
    (ngrams_path / "info.json").write_text('{"ngram_size": 1}', encoding="utf-8")

    docs = {("task", "set"): ["hello world"]}
    assert get_train_overlap(docs, str(ngrams_path), limit=10) == {"task": set()}

    lookup_path = tmp_path / "data/task/set_1grams_limit10.lookup.json"
    overlap_path = tmp_path / "data/task/set_1grams_limit10.overlaps.json"
    assert json.loads(lookup_path.read_text(encoding="utf-8")) == {
        "hello": [0],
        "world": [0],
    }
    assert json.loads(overlap_path.read_text(encoding="utf-8")) == []

    assert get_train_overlap(docs, str(ngrams_path), limit=10) == {"task": set()}


def test_cache_loaders_validate_json_types(tmp_path):
    overlap_path = tmp_path / "overlaps.json"
    overlap_path.write_text('[0, "one"]', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid overlap cache format"):
        _load_overlap_cache(str(overlap_path))

    lookup_path = tmp_path / "lookup.json"
    lookup_path.write_text('{"hello": [0, "one"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid lookup cache format"):
        _load_lookup_cache(str(lookup_path))
