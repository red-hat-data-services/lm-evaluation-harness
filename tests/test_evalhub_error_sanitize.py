"""Unit tests for Eval Hub adapter error sanitization (main._sanitize_error_message)."""

import pytest
import requests

from main import _evaluation_failure_for_evalhub, _sanitize_error_message


@pytest.mark.parametrize(
    ("raw", "forbidden", "must_contain"),
    [
        (
            "401 for url: https://example.com/path?token=SECRET&ref=1",
            ["SECRET", "token=SECRET"],
            "https://example.com/path",
        ),
        (
            "fail https://u:p@evil.test/path",
            ["u:p@", "p@", "u:p"],
            "https://evil.test/path",
        ),
        (
            "see https://huggingface.co/m/model#section",
            ["#section"],
            "https://huggingface.co/m/model",
        ),
        (
            "Authorization: Bearer eyJhbGciOiJFAKE",
            ["eyJhbG", "FAKE"],
            "[redacted]",
        ),
        (
            "oops token=abc123xyz trailing",
            ["abc123xyz"],
            "token=[redacted]",
        ),
        (
            "access_token=sekret",
            ["sekret"],
            "access_token=[redacted]",
        ),
        (
            "OAuth error client_secret=ABC123 end",
            ["ABC123", "client_secret=ABC"],
            "client_secret=[redacted]",
        ),
        (
            "Error:api_key=XYZZY",
            ["XYZZY"],
            "api_key=[redacted]",
        ),
        (
            "msg, password=hunter2 tail",
            ["hunter2"],
            "password=[redacted]",
        ),
        (
            "refresh_token=R1\nnext line",
            ["R1"],
            "refresh_token=[redacted]",
        ),
        (
            'API body {"client_secret":"ABC123"}',
            ["ABC123"],
            '"client_secret":"[redacted]"',
        ),
        (
            '{"access_token": "sekret"}',
            ["sekret"],
            '"access_token":"[redacted]"',
        ),
        (
            "{'password': 'hunter2'}",
            ["hunter2"],
            "'password':'[redacted]'",
        ),
    ],
)
def test_sanitize_removes_secrets(raw: str, forbidden: list[str], must_contain: str) -> None:
    out = _sanitize_error_message(raw)
    for s in forbidden:
        assert s not in out, out
    assert must_contain in out, out


def test_sanitize_preserves_plain_urls() -> None:
    msg = "401 Client Error: Unauthorized for url: https://api.example.com/v1/completions"
    assert _sanitize_error_message(msg) == msg


def test_no_false_positive_inside_identifier() -> None:
    """Do not redact when 'token' (or key names) appear only inside a larger word."""
    for msg in (
        "mytokenname",
        "unknown_tokendriver",
        "error: myaccess_tokenish_value",
    ):
        assert _sanitize_error_message(msg) == msg, msg


# Sample Hub tracebacks (any gated *model* repo, not Llama-specific). EvalHub's
# user-facing message is generic; Hub itself names whichever tokenizer/dataset failed.
_TOKENIZER_GATE_MSG = """\
OSError: You are trying to access a gated repo.
Make sure to have access to it at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct.
401 Client Error.

Cannot access gated repo for url https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/config.json.
Access to model meta-llama/Llama-3.1-8B-Instruct is restricted. You must have access to it and be authenticated to access it. Please log in.
"""

_TOKENIZER_GATE_MSG_OTHER_MODEL = """\
Cannot access gated repo for url https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/resolve/main/config.json.
Access to model mistralai/Mistral-7B-Instruct-v0.2 is restricted.
"""

_DATASET_GATE_MSG = """\
Dataset 'cais/hle' is a gated dataset on the Hub.
Cannot access gated repo for url https://huggingface.co/api/datasets/cais/hle.
Access to dataset cais/hle is restricted.
"""


def test_gated_tokenizer_error_is_not_labeled_dataset() -> None:
    """RHOAIENG-95264: AutoTokenizer.from_pretrained on a gated model repo."""
    msg, code = _evaluation_failure_for_evalhub(OSError(_TOKENIZER_GATE_MSG))
    assert "tokenizer" in msg.lower()
    assert "dataset" not in msg.lower()
    assert "accessible tokenizer" in msg
    assert "hf-token" in msg
    assert code == "gated_tokenizer_auth_required"


def test_gated_tokenizer_error_not_tied_to_one_model_id() -> None:
    msg, code = _evaluation_failure_for_evalhub(OSError(_TOKENIZER_GATE_MSG_OTHER_MODEL))
    assert code == "gated_tokenizer_auth_required"
    assert "tokenizer" in msg.lower()


def test_gated_dataset_error_keeps_dataset_wording() -> None:
    msg, code = _evaluation_failure_for_evalhub(OSError(_DATASET_GATE_MSG))
    assert "dataset" in msg.lower()
    assert "tokenizer" not in msg.lower()
    assert code == "gated_dataset_auth_required"


def test_gated_tokenizer_detected_from_exception_cause() -> None:
    wrapped = RuntimeError("Evaluation failed")
    wrapped.__cause__ = OSError(_TOKENIZER_GATE_MSG)
    msg, code = _evaluation_failure_for_evalhub(wrapped)
    assert code == "gated_tokenizer_auth_required"
    assert "tokenizer" in msg.lower()


def test_unspecified_gated_repo_does_not_claim_dataset() -> None:
    msg, code = _evaluation_failure_for_evalhub(OSError("You are trying to access a gated repo."))
    assert "dataset" not in msg.lower()
    assert "tokenizer" not in msg.lower()
    assert "resource" in msg.lower()
    assert code == "gated_hf_auth_required"


class _FakeResponse:
    def __init__(self, url: str, status_code: int = 403) -> None:
        self.url = url
        self.status_code = status_code


def test_hub_403_is_gated_hf() -> None:
    err = requests.HTTPError("403 Client Error")
    err.response = _FakeResponse("https://huggingface.co/api/models/org/gated")
    _, code = _evaluation_failure_for_evalhub(err)
    assert code == "gated_hf_auth_required"


def test_inference_endpoint_403_is_model_access_forbidden() -> None:
    url = "https://xyz.us-east-1.aws.endpoints.huggingface.cloud/v1/completions"
    err = requests.HTTPError(f"403 Client Error: Forbidden for url: {url}")
    err.response = _FakeResponse(url)
    msg, code = _evaluation_failure_for_evalhub(err)
    assert code == "model_access_forbidden"
    assert msg.startswith("Model endpoint returned HTTP 403")


