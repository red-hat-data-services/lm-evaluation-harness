"""Chat-template opt-in and text completions regression coverage."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast

import main
from lm_eval.models.api_models import JsonChatStr
from lm_eval.models.openai_completions import LocalChatCompletion, LocalCompletionsAPI


@pytest.mark.parametrize("enabled", [False, True])
def test_adapter_passes_chat_template_to_evaluator(monkeypatch, tmp_path, enabled):
    import datasets

    # The image uses datasets 3.x; keep this unit test compatible with newer dev environments.
    monkeypatch.setattr(
        datasets.config, "HF_DATASETS_TRUST_REMOTE_CODE", False, raising=False
    )
    monkeypatch.setattr(main, "__file__", str(tmp_path / "main.py"))
    monkeypatch.setattr(
        main, "resolve_model_credentials", lambda: SimpleNamespace(api_key=None)
    )
    monkeypatch.setattr(main, "read_model_auth_key", lambda key: None)
    monkeypatch.setattr(main, "TaskManager", MagicMock())
    monkeypatch.setattr(
        main, "_resolve_lmeval_task", lambda benchmark, manager: benchmark
    )
    monkeypatch.setattr(main, "_build_additional_info", lambda **kwargs: {})
    evaluate = MagicMock(
        return_value={
            "results": {"ifeval": {"prompt_level_strict_acc,none": 1.0}},
            "samples": {"ifeval": [{}]},
        }
    )
    monkeypatch.setattr(main, "simple_evaluate", evaluate)
    config = SimpleNamespace(
        id="00000000-0000-0000-0000-000000000001",
        benchmark_id="ifeval",
        benchmark_index=0,
        num_examples=1,
        model=SimpleNamespace(name="test/model", url="http://localhost:8000/v1"),
        parameters={"apply_chat_template": enabled},
        exports=None,
        model_dump=lambda: {},
    )
    adapter = main.LMEvalAdapter.__new__(main.LMEvalAdapter)
    result = adapter.run_benchmark_job(config, MagicMock())
    assert result.overall_score == 1.0
    assert evaluate.call_args.kwargs["apply_chat_template"] is enabled
    assert evaluate.call_args.kwargs["model_args"]["tokenized_requests"] is False


def test_chat_template_default_preserves_raw_prompt_mode():
    assert main._chat_template_enabled({}) is False


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, {}])
def test_chat_template_rejects_non_boolean(value):
    with pytest.raises(ValueError, match="apply_chat_template must be a boolean"):
        main._chat_template_enabled({"apply_chat_template": value})


@pytest.fixture
def text_api(monkeypatch):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        ["<user>Answer Yes or No.[EOS]<assistant>Yes No"],
        vocab_size=300,
        special_tokens=["[UNK]", "[EOS]"],
    )
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        unk_token="[UNK]",
        eos_token="[EOS]",
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}<{{ message['role'] }}>"
        "{{ message['content'] }}[EOS]{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: hf_tokenizer,
    )
    return LocalCompletionsAPI(
        model="test/model",
        base_url="http://localhost:8000/v1/completions",
        tokenizer_backend="huggingface",
        tokenized_requests=False,
    )


def test_text_template_supports_generation_and_choice_scoring(text_api, monkeypatch):
    history = [{"role": "user", "content": "Answer Yes or No."}]
    prompt = text_api.apply_chat_template(history)
    assert prompt == "<user>Answer Yes or No.[EOS]<assistant>"
    assert isinstance(prompt, str)
    assert not isinstance(prompt, JsonChatStr)
    context, answer = text_api._encode_pair(prompt, " Yes")
    assert context and answer
    captured = []

    def post(url, **kwargs):
        captured.append(kwargs["json"])
        return SimpleNamespace(
            ok=True,
            raise_for_status=lambda: None,
            json=lambda: {"choices": [{"index": 0, "text": "Yes"}]},
        )

    monkeypatch.setattr("requests.post", post)
    text_api.model_call([prompt], generate=True, gen_kwargs={"max_gen_toks": 2048})
    text_api.model_call([context + answer], generate=False)
    assert captured[0]["prompt"] == prompt
    assert captured[0]["max_tokens"] == 2048
    assert "[EOS]" in captured[0]["stop"]
    assert isinstance(captured[1]["prompt"], str)
    assert captured[1]["echo"] is True
    assert captured[1]["max_tokens"] == 1
    assert captured[1]["logprobs"] == 1


def test_chat_endpoint_preserves_message_payload(text_api):
    api = LocalChatCompletion.__new__(LocalChatCompletion)
    api.tokenizer_backend = "huggingface"
    api.tokenizer = text_api.tokenizer
    api.tokenized_requests = False
    api._batch_size = 1
    history = [{"role": "user", "content": "Answer Yes or No."}]
    prompt = api.apply_chat_template(history)
    assert isinstance(prompt, JsonChatStr)
    assert api.create_message([prompt]) == history


def test_token_id_mode_keeps_same_rendered_template(text_api):
    history = [{"role": "user", "content": "Yes"}]
    expected = text_api.apply_chat_template(history)
    text_api.tokenized_requests = True
    assert text_api.apply_chat_template(history) == expected


def test_template_can_continue_final_assistant_message(text_api):
    prompt = text_api.apply_chat_template(
        [{"role": "assistant", "content": "Yes"}],
        add_generation_prompt=False,
    )
    assert prompt == "<assistant>Yes"


@pytest.mark.parametrize("output_type", ["generate_until", "multiple_choice"])
def test_harness_evaluation_sends_rendered_text(text_api, monkeypatch, output_type):
    """Exercise task construction, templating and the API path together without downloads."""
    from datasets import Dataset, DatasetDict

    from lm_eval.api.task import ConfigurableTask
    from lm_eval.evaluator import simple_evaluate

    dataset = DatasetDict(
        {
            "test": Dataset.from_list(
                [{"question": "Answer Yes or No.", "answer": "Yes"}]
            )
        }
    )
    monkeypatch.setattr("datasets.load_dataset", lambda *args, **kwargs: dataset)
    captured = []

    def post(url, **kwargs):
        payload = kwargs["json"]
        captured.append(payload)
        choice = {"index": 0, "text": "Yes"}
        if payload.get("echo"):
            tokens = text_api.tok_encode(payload["prompt"])
            choice["logprobs"] = {
                "token_logprobs": [-1.0] * (len(tokens) + 1),
                "top_logprobs": [{"Yes": -1.0}] * (len(tokens) + 1),
            }
        return SimpleNamespace(
            ok=True, raise_for_status=lambda: None, json=lambda: {"choices": [choice]}
        )

    monkeypatch.setattr("requests.post", post)
    config = {
        "task": "test_chat_template",
        "dataset_path": "fixture",
        "test_split": "test",
        "doc_to_text": "question",
        "output_type": output_type,
        "metric_list": [
            {"metric": "acc", "aggregation": "mean", "higher_is_better": True}
        ],
    }
    if output_type == "multiple_choice":
        config.update(doc_to_choice=["Yes", "No"], doc_to_target=0)
    else:
        config.update(
            doc_to_target="answer",
            generation_kwargs={"max_gen_toks": 2048, "temperature": 0},
            process_results=lambda doc, responses: {
                "acc": float(responses[0] == doc["answer"])
            },
        )
    task = ConfigurableTask(config=config)
    results = simple_evaluate(
        model=text_api,
        tasks=[task],
        num_fewshot=0,
        apply_chat_template=True,
        limit=1,
        bootstrap_iters=0,
        log_samples=True,
    )
    assert results["results"]["test_chat_template"]["acc,none"] == 1.0
    assert len(captured) == (2 if output_type == "multiple_choice" else 1)
    for payload in captured:
        assert isinstance(payload["prompt"], str)
        assert payload["prompt"].startswith("<user>Answer Yes or No.[EOS]<assistant>")
