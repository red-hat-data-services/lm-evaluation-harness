"""Shared JSONSchemaBench format scoring regressions."""

import json

import pytest
from jsonschema.exceptions import FormatError

from lm_eval.tasks.jsonschema_bench.metrics import format_checker, schema_compliance


@pytest.mark.parametrize(
    ("fmt", "value"),
    [
        ("uuid", "01234567-89ab-cdef-0123-456789abcdef"),
        ("uuid", "00000000-0000-0000-0000-000000000000"),
        ("ipv4", "192.168.1.1"),
        ("ipv4", "0.0.0.0"),
        ("ipv6", "2001:db8::1"),
        ("ipv6", "::1"),
    ],
)
def test_valid_formats_pass_scoring(fmt, value):
    format_checker.check(value, fmt)
    schema = json.dumps({"type": "string", "format": fmt})
    assert schema_compliance([schema], [json.dumps(value)])


@pytest.mark.parametrize(
    ("fmt", "value"),
    [
        ("uuid", "not-a-uuid"),
        ("uuid", "0123456789abcdef0123456789abcdef"),
        ("ipv4", "999.168.1.1"),
        ("ipv4", "192.168.1"),
        ("ipv6", "2001:db8::g"),
        ("ipv6", "192.168.1.1"),
        ("ipv6", "fe80::1%eth0"),
    ],
)
def test_invalid_formats_fail_scoring(fmt, value):
    with pytest.raises(FormatError):
        format_checker.check(value, fmt)
    schema = json.dumps({"type": "string", "format": fmt})
    assert not schema_compliance([schema], [json.dumps(value)])


@pytest.mark.parametrize("fmt", ["uuid", "ipv4", "ipv6"])
@pytest.mark.parametrize("value", [None, 123, {}])
def test_format_only_applies_to_strings_and_type_checks_still_apply(fmt, value):
    instance = json.dumps({"value": value})
    schema = {"type": "object", "properties": {"value": {"format": fmt}}}
    assert schema_compliance([json.dumps(schema)], [instance])
    schema["properties"]["value"]["type"] = "string"
    assert not schema_compliance([json.dumps(schema)], [instance])
