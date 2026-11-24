"""Perplexity must not initialize CAREQA's generative metrics."""

import importlib.util
import math
import socket
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import datasets
import evaluate
import pytest

from lm_eval.evaluator import simple_evaluate
from lm_eval.models.dummy import DummyLM
from lm_eval.utils import load_yaml_config


TASK_DIR = Path(__file__).parents[1] / "lm_eval" / "tasks" / "careqa"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("CAREQA regression tests must not access the network")

    monkeypatch.setattr(socket.socket, "connect", fail)
    monkeypatch.setattr(socket, "create_connection", fail)


def load_utils():
    spec = importlib.util.spec_from_file_location(
        "careqa_utils_under_test", TASK_DIR / "utils_open.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_loader(monkeypatch, *, missing_bleurt=False):
    values = {
        "bleu": {"bleu": 0.5},
        "rouge": {"rouge1": 0.1, "rouge2": 0.2, "rougeL": 0.3},
        "bertscore": {"f1": [0.6]},
        "bleurt": {"scores": [0.7]},
    }

    def load(name, *args, **kwargs):
        if name == "bleurt" and missing_bleurt:
            raise ImportError("Missing dependency: bleurt")
        return SimpleNamespace(compute=MagicMock(return_value=values[name]))

    loader = MagicMock(side_effect=load)
    monkeypatch.setattr(evaluate, "load", loader)
    return loader


def test_document_helpers_do_not_initialize_metrics(monkeypatch):
    loader = metric_loader(monkeypatch, missing_bleurt=True)
    utils = load_utils()
    doc = {"question": "A question", "answer": "An answer"}
    assert utils.doc_to_text(doc) == "A question"
    assert utils.doc_to_target(doc) == "An answer"
    loader.assert_not_called()


def test_perplexity_yaml_inheritance_does_not_initialize_metrics(monkeypatch):
    loader = metric_loader(monkeypatch, missing_bleurt=True)
    config = load_yaml_config(str(TASK_DIR / "careqa_open_perplexity.yaml"))
    assert config["output_type"] == "loglikelihood_rolling"
    assert config["doc_to_target"]({"answer": "text"}) == "text"
    assert {m["metric"] for m in config["metric_list"]} == {
        "word_perplexity",
        "byte_perplexity",
        "bits_per_byte",
    }
    loader.assert_not_called()


def test_one_example_perplexity_returns_actual_likelihood_metrics(monkeypatch):
    loader = metric_loader(monkeypatch, missing_bleurt=True)
    answer = "short answer"
    local_data = datasets.DatasetDict(
        {
            "test": datasets.Dataset.from_list(
                [
                    {"question": "Question", "answer": answer},
                    {"question": "Not selected", "answer": "Not evaluated"},
                ]
            )
        }
    )
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: local_data)

    class RecordingLM(DummyLM):
        def __init__(self):
            super().__init__()
            self.seen = []

        def loglikelihood_rolling(self, requests, disable_tqdm=False):
            self.seen.extend(request.args[0] for request in requests)
            return [-2.0] * len(requests)

    model = RecordingLM()
    results = simple_evaluate(
        model=model,
        tasks=["careqa_open_perplexity"],
        limit=1,
        bootstrap_iters=0,
        log_samples=True,
    )
    scores = results["results"]["careqa_open_perplexity"]
    assert scores["word_perplexity,none"] == pytest.approx(math.e)
    assert scores["byte_perplexity,none"] == pytest.approx(math.exp(2 / len(answer)))
    assert scores["bits_per_byte,none"] == pytest.approx(
        2 / (len(answer) * math.log(2))
    )
    assert model.seen == [answer]
    assert len(results["samples"]["careqa_open_perplexity"]) == 1
    loader.assert_not_called()


def test_generative_metrics_are_preserved_and_loaded_once(monkeypatch):
    loader = metric_loader(monkeypatch)
    utils = load_utils()
    loader.assert_not_called()
    expected = {
        "bleu": 0.5,
        "rouge1": 0.1,
        "rouge2": 0.2,
        "rougeL": 0.3,
        "bleurt": 0.7,
        "bert_score": 0.6,
    }
    for _ in range(2):
        assert (
            utils.process_results_gen({"answer": "reference"}, ["prediction"])
            == expected
        )
    assert [c.args[0] for c in loader.call_args_list] == [
        "bleu",
        "rouge",
        "bertscore",
        "bleurt",
    ]
    assert loader.call_args_list[-1].args == ("bleurt", "bleurt-base-512")
    assert loader.call_args_list[-1].kwargs == {"module_type": "metric"}


def test_missing_bleurt_is_identified_when_generation_needs_it(monkeypatch):
    metric_loader(monkeypatch, missing_bleurt=True)
    utils = load_utils()
    with pytest.raises(RuntimeError, match="bleurt") as error:
        utils.process_results_gen({"answer": "reference"}, ["prediction"])
    assert isinstance(error.value.__cause__, ImportError)
    assert "Missing dependency: bleurt" in str(error.value)


def test_metric_loading_failure_remains_an_error(monkeypatch):
    cause = ConnectionError("fixture metric source unavailable")
    loader = MagicMock(side_effect=cause)
    monkeypatch.setattr(evaluate, "load", loader)
    utils = load_utils()
    with pytest.raises(RuntimeError, match="bleu") as error:
        utils.process_results_gen({"answer": "reference"}, ["prediction"])
    assert error.value.__cause__ is cause
