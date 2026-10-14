"""JSONSchemaBench Easy generation and strict scoring regressions."""

from copy import deepcopy
from pathlib import Path

import pytest

from lm_eval.api.task import TaskConfig
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.tasks.jsonschema_bench.metrics import json_validity, schema_compliance
from lm_eval.utils import load_yaml_config


@pytest.fixture
def task_config():
    path = (
        Path(__file__).resolve().parents[1]
        / "lm_eval/tasks/jsonschema_bench/jsonschema_bench_easy.yaml"
    )
    return TaskConfig(**load_yaml_config(str(path)))


def test_pretty_json_is_not_cut_at_blank_lines(task_config):
    api = LocalCompletionsAPI(
        base_url="http://unused.invalid/v1/completions",
        model="test-model",
        tokenizer_backend=None,
    )
    payload = api._create_payload(
        ["prompt"],
        generate=True,
        gen_kwargs=deepcopy(task_config.generation_kwargs),
        eos="<|eot_id|>",
    )
    response = '{\n    "name": "Theme",\n\n    "body": []\n}'
    # Simulate the endpoint obeying the actual stop list emitted by the backend.
    for stop in payload["stop"]:
        response = response.split(stop, 1)[0]
    assert json_validity(["{}"], [response])
    assert schema_compliance(
        ['{"type":"object","required":["name","body"]}'], [response]
    )
    assert payload["stop"] == ["<|eot_id|>"]
    assert payload["max_tokens"] == 256  # Output remains bounded.


@pytest.mark.parametrize(
    "response",
    ['Here is the JSON:\n\n{"name":"Theme"}', '{"name":"Theme"', ""],
)
def test_explanations_and_incomplete_json_remain_invalid(response):
    assert not json_validity(["{}"], [response])
    assert not schema_compliance(["{}"], [response])


@pytest.mark.parametrize("split", ["medium", "hard"])
def test_other_splits_inherit_prompt_and_stopping_policy(split, task_config):
    path = (
        Path(__file__).resolve().parents[1]
        / f"lm_eval/tasks/jsonschema_bench/jsonschema_bench_{split}.yaml"
    )
    config = TaskConfig(**load_yaml_config(str(path)))
    assert config.description == task_config.description
    assert config.generation_kwargs == {"until": [], "do_sample": False}
    assert config.metadata["version"] == 0.2
