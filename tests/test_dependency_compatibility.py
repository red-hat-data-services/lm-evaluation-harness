"""Offline regressions for the datasets 5 / huggingface-hub 1 image stack."""

from pathlib import Path


def test_unitxt_import_and_public_entrypoints():
    # Unitxt 1.17.2 imports datasets.utils.py_utils.get_imports (removed in v4).
    from unitxt import evaluate, load_dataset

    assert callable(load_dataset)
    assert callable(evaluate)


def test_unitxt_loads_in_memory_recipe():
    from unitxt import load_dataset
    from unitxt.card import TaskCard
    from unitxt.loaders import LoadFromDictionary
    from unitxt.task import Task
    from unitxt.templates import InputOutputTemplate

    card = TaskCard(
        loader=LoadFromDictionary(
            data={"test": [{"question": "What is 2+2?", "answer": "4"}]},
            data_classification_policy=["public"],
        ),
        task=Task(
            input_fields={"question": str},
            reference_fields={"answer": str},
            prediction_type=str,
            metrics=["metrics.accuracy"],
        ),
    )
    dataset = load_dataset(
        card=card,
        template=InputOutputTemplate(
            input_format="{question}", output_format="{answer}"
        ),
    )
    assert dataset["test"][0]["source"].strip() == "What is 2+2?"
    assert dataset["test"][0]["target"] == "4"


def test_code_eval_metric_loads():
    import evaluate

    # Load the repository's metric without network or model-generated code execution.
    metric_path = Path(__file__).resolve().parents[1] / "metrics" / "code_eval"
    metric = evaluate.load(str(metric_path))
    assert metric.name == "code_eval"


def test_evaluate_hub_authentication(monkeypatch, tmp_path):
    # evaluate 0.4.3 uses the removed hf_api.HfFolder, even without a token.
    import huggingface_hub.constants
    from evaluate.utils.file_utils import get_authentication_headers_for_url

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub.constants, "HF_TOKEN_PATH", str(tmp_path / "token"))
    url = "https://huggingface.co/spaces/evaluate-metric/code_eval/resolve/main/code_eval.py"
    assert "authorization" not in get_authentication_headers_for_url(url, token=False)
    assert "authorization" not in get_authentication_headers_for_url(url)
    headers = get_authentication_headers_for_url(url, token="test-token")
    assert headers["authorization"] == "Bearer test-token"
