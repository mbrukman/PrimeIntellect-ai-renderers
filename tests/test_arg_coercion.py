"""Schema-aware argument coercion parity with vLLM / SGLang.

The XML-style tool-call wire format (Qwen3.5, GLM, MiniMax, Laguna)
renders parameter values verbatim with no quoting, so the parser must
consult the tool schema to know whether ``True`` is a bool or a string.
Renderers' previous coercion was strict ``json.loads``, which flagged
``True`` / ``False`` (capitalised Python literals) as ``INVALID_JSON``
for boolean params — but both vLLM's ``Qwen3CoderToolParser`` and
SGLang's ``Qwen3CoderDetector`` accept them via case-folded comparison.

These tests pin the new ladder to the vLLM / SGLang reference shape and
guard the bug from issue #47.
"""

from __future__ import annotations

import pytest

from renderers.base import ToolCallParseStatus, load_tokenizer
from renderers.parsing import _coerce_arg_value, parse_qwen35


# ── Direct coercion ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("True", True),
        ("TRUE", True),
        ("false", False),
        ("False", False),
        ("FALSE", False),
    ],
)
def test_boolean_accepts_case_insensitive(raw, expected):
    value, used_fallback = _coerce_arg_value(raw, {"type": "boolean"})
    assert value is expected
    assert used_fallback is False


@pytest.mark.parametrize("declared", ["boolean", "bool", "binary"])
def test_boolean_type_aliases_all_accepted(declared):
    value, used_fallback = _coerce_arg_value("True", {"type": declared})
    assert value is True
    assert used_fallback is False


def test_boolean_garbage_degenerates_to_false_with_invalid_flag():
    """Match vLLM: non-true/false values silently become ``False`` — but we
    still flag the fallback so verifier / RL-loss code can see the drift."""
    value, used_fallback = _coerce_arg_value("yes", {"type": "boolean"})
    assert value is False
    assert used_fallback is True


@pytest.mark.parametrize("declared", ["boolean", "integer", "object", "number"])
def test_null_literal_is_case_insensitive_for_non_string_types(declared):
    for raw in ("null", "Null", "NULL"):
        value, used_fallback = _coerce_arg_value(raw, {"type": declared})
        assert value is None
        assert used_fallback is False


def test_null_is_preserved_verbatim_for_string_type():
    """Deliberate deviation from vLLM/SGLang: string-typed ``"null"``
    stays as the string ``"null"`` so the existing string-verbatim
    contract holds (see ``test_tool_arg_type_preservation``). The XML
    wire format already can't distinguish the string ``"null"`` from
    JSON null, but when a schema says ``type: "string"`` we honour it."""
    value, used_fallback = _coerce_arg_value("null", {"type": "string"})
    assert value == "null"
    assert used_fallback is False


@pytest.mark.parametrize(
    "declared,raw,expected",
    [
        ("integer", "42", 42),
        ("int", "-7", -7),
        ("uint", "0", 0),
        ("long", "9999999999", 9999999999),
        ("short", "12", 12),
        ("unsigned", "5", 5),
    ],
)
def test_int_family_coerces(declared, raw, expected):
    value, used_fallback = _coerce_arg_value(raw, {"type": declared})
    assert value == expected
    assert isinstance(value, int)
    assert used_fallback is False


def test_int_failure_keeps_raw_and_flags_fallback():
    value, used_fallback = _coerce_arg_value("abc", {"type": "integer"})
    assert value == "abc"
    assert used_fallback is True


@pytest.mark.parametrize(
    "raw,expected,exact_type",
    [
        ("3.14", 3.14, float),
        ("1e3", 1000.0, float),  # source has `e` → stays float (SGLang rule)
        ("1.0", 1.0, float),  # source has `.` → stays float
        ("-2.5", -2.5, float),
        ("7", 7, int),  # no `.`/`e` and whole → demoted to int
    ],
)
def test_float_family_coerces(raw, expected, exact_type):
    value, used_fallback = _coerce_arg_value(raw, {"type": "number"})
    assert value == expected
    assert type(value) is exact_type
    assert used_fallback is False


def test_float_failure_keeps_raw_and_flags_fallback():
    value, used_fallback = _coerce_arg_value("not-a-number", {"type": "float"})
    assert value == "not-a-number"
    assert used_fallback is True


def test_object_via_json_loads():
    value, used_fallback = _coerce_arg_value('{"k": 1}', {"type": "object"})
    assert value == {"k": 1}
    assert used_fallback is False


def test_object_via_ast_literal_eval_fallback():
    """Python-literal dicts (single quotes) should still parse for object
    params — this is the vLLM ``ast.literal_eval`` fallback path."""
    value, used_fallback = _coerce_arg_value("{'k': 1}", {"type": "object"})
    assert value == {"k": 1}
    assert used_fallback is False


def test_array_via_json_loads():
    value, used_fallback = _coerce_arg_value("[1, 2, 3]", {"type": "array"})
    assert value == [1, 2, 3]
    assert used_fallback is False


def test_object_total_failure_flags_invalid():
    value, used_fallback = _coerce_arg_value(
        "this is not a json object", {"type": "object"}
    )
    assert value == "this is not a json object"
    assert used_fallback is True


def test_anyof_treated_as_object():
    """vLLM-specific: a schema with ``anyOf`` and no top-level ``type``
    routes through the object branch so JSON-shaped values parse."""
    schema = {"anyOf": [{"type": "integer"}, {"type": "string"}]}
    value, used_fallback = _coerce_arg_value("42", schema)
    assert value == 42
    assert used_fallback is False


@pytest.mark.parametrize("declared", ["string", "str", "text", "varchar", "char", "enum"])
def test_string_family_returns_verbatim(declared):
    value, used_fallback = _coerce_arg_value("True", {"type": declared})
    assert value == "True"
    assert used_fallback is False


def test_string_with_list_form_type():
    value, used_fallback = _coerce_arg_value("True", {"type": ["string"]})
    assert value == "True"
    assert used_fallback is False


def test_no_schema_falls_back_to_json_loads():
    """Historical behavior: when ``param_schema`` is ``None`` (no tools
    were passed), we still try ``json.loads`` for backwards compatibility
    so untyped numbers / bools parse. ``None`` is handled via the
    ``null`` short-circuit at the top of the ladder."""
    value, used_fallback = _coerce_arg_value("42", None)
    assert value == 42
    assert used_fallback is False

    value, used_fallback = _coerce_arg_value("hello", None)
    assert value == "hello"
    assert used_fallback is True


def test_unknown_type_falls_back_to_literal_eval():
    """A declared type we don't recognise (e.g. ``"date"``) lands in the
    catch-all that tries ``ast.literal_eval``, mirroring vLLM's else
    branch — strings without quotes that fail to eval stay as strings."""
    value, used_fallback = _coerce_arg_value("2024-01-01", {"type": "date"})
    assert value == "2024-01-01"
    assert used_fallback is True

    # Bare Python literal still parses through the catch-all
    value, used_fallback = _coerce_arg_value("[1, 2]", {"type": "date"})
    assert value == [1, 2]
    assert used_fallback is False


# ── Integration: parse_qwen35 end-to-end (the bug from issue #47) ───


_TOOLS_FROM_ISSUE = [
    {
        "type": "function",
        "function": {
            "name": "filesystem_server_get_directory_tree",
            "description": "List a directory tree.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer"},
                    "include_files": {"type": "boolean"},
                    "show_size": {"type": "boolean"},
                },
            },
        },
    }
]


def test_qwen35_capital_true_for_boolean_is_ok_status():
    """Regression for issue #47: ``<parameter=include_files>True</parameter>``
    used to be flagged ``INVALID_JSON`` because ``json.loads("True")``
    fails. SGLang and vLLM both accept it via case-folded comparison;
    renderers now does too."""
    tok = load_tokenizer("Qwen/Qwen3.5-9B")

    completion = (
        "<tool_call>\n"
        "<function=filesystem_server_get_directory_tree>\n"
        "<parameter=path>/</parameter>\n"
        "<parameter=max_depth>2</parameter>\n"
        "<parameter=include_files>True</parameter>\n"
        "<parameter=show_size>True</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    ids = tok.encode(completion, add_special_tokens=False)

    parsed = parse_qwen35(
        tok,
        list(ids),
        stop_ids={tok.eos_token_id} if tok.eos_token_id is not None else set(),
        think_id=tok.convert_tokens_to_ids("<think>"),
        think_end_id=tok.convert_tokens_to_ids("</think>"),
        tool_call_id=tok.convert_tokens_to_ids("<tool_call>"),
        tool_call_end_id=tok.convert_tokens_to_ids("</tool_call>"),
        tools=_TOOLS_FROM_ISSUE,
    )
    assert len(parsed.tool_calls) == 1
    tc = parsed.tool_calls[0]
    assert tc.status == ToolCallParseStatus.OK
    assert tc.name == "filesystem_server_get_directory_tree"
    assert tc.arguments == {
        "path": "/",
        "max_depth": 2,
        "include_files": True,
        "show_size": True,
    }
