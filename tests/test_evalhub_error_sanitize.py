"""Unit tests for Eval Hub adapter error sanitization (main._sanitize_error_message)."""

import pytest

from main import _sanitize_error_message


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

