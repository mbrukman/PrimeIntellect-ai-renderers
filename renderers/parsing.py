"""Token-level parsing — operates on token IDs directly.

Finds special token boundaries by scanning token IDs, then decodes only
the text segments between them. No regex on decoded text, no false positives
from content that happens to look like special tokens.

Every parser emits ``list[ParsedToolCall]`` covering every attempt —
successful and malformed alike — with a ``status`` enum classifying the
outcome and a ``token_span`` recording where in the (stop-stripped)
token stream the attempt sat. Callers filter on ``status == OK`` for the
clean subset; verifier and RL-loss code uses the rest. This diverges from
vLLM's ``ExtractedToolCallInformation`` (single ``tools_called`` bool, no
per-call status) and SGLang's ``StreamingParseResult`` (silent drop on
failure) — see ``ToolCallParseStatus`` docstring for the rationale.
"""

from __future__ import annotations

import ast
import json
import re
import uuid
from typing import Any

from renderers.base import ParsedResponse, ParsedToolCall, ToolCallParseStatus, ToolSpec


# ── vLLM Qwen3CoderToolParser regex patterns ────────────────────────
#
# Lifted verbatim from ``vllm/tool_parsers/qwen3coder_tool_parser.py``
# (``self.tool_call_regex`` / ``tool_call_function_regex`` /
# ``tool_call_parameter_regex``). Each pattern has two branches: one
# for the closed form (with the matching end tag) and one anchored at
# end-of-string for unclosed output. The parameter regex uses
# positive lookaheads so a missing ``</parameter>`` is recovered from
# the next ``<parameter=`` / ``</function>`` boundary instead of
# silently dropping the value.


# Mask matching vLLM's ``vllm.utils.MASK_64_BITS`` so the random id we
# generate is byte-shape compatible with vLLM's ``random_uuid()``.
_MASK_64_BITS = (1 << 64) - 1


def _make_tool_call_id() -> str:
    """Mint a tool-call id in vLLM's default format.

    vLLM's ``ToolCall`` dataclass uses ``default_factory=make_tool_call_id``
    which returns ``f"chatcmpl-tool-{random_uuid()}"`` where
    ``random_uuid`` is ``f"{uuid.uuid4().int & MASK_64_BITS:016x}"`` —
    16 hex characters drawn from the low 64 bits of a v4 UUID. We
    reproduce the format so OpenAI-shaped clients get the same id
    surface across renderers and vLLM serving paths.
    """
    return f"chatcmpl-tool-{uuid.uuid4().int & _MASK_64_BITS:016x}"


_TOOL_CALL_BLOCK_RE = re.compile(
    r"<tool_call>(.*?)</tool_call>|<tool_call>(.*?)$", re.DOTALL
)
_FUNCTION_BLOCK_RE = re.compile(
    r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL
)
_PARAMETER_BLOCK_RE = re.compile(
    r"<parameter=(.*?)(?:</parameter>|(?=<parameter=)|(?=</function>)|$)",
    re.DOTALL,
)


# ── Schema-aware argument coercion ──────────────────────────────────
#
# XML-style tool-call formats render argument values verbatim inside
# ``<parameter=…>`` tags with no quoting. ``true`` and the string
# ``"true"`` produce identical wire bytes; without the tool schema, the
# parser has no signal to distinguish them. We mirror vLLM's
# ``Qwen3CoderToolParser`` (and SGLang's ``Qwen3CoderDetector``) end to
# end: case-insensitive ``null`` short-circuit, ``int()`` / ``float()``
# for numeric types, ``.lower() == "true"`` for bool, ``json.loads``
# with ``ast.literal_eval`` fallback for objects/arrays. That accepts
# both JSON literals (Qwen3.6) and Python literals (Qwen3.5 — ``str()``
# renders bools as ``True`` / ``False``) on the inference side.
#
# Why ``qwen3_coder`` and not ``qwen3_xml``: vLLM ships two parsers for
# the same wire format. ``qwen3_xml`` is the newer streaming-first
# parser (uses ``xml.parsers.expat`` and is the recommended choice for
# *live serving* because ``qwen3_coder`` has known infinite-loop bugs
# in vLLM's streaming path). For offline token-level extraction the
# two parsers agree on JSON-literal scalars but ``qwen3_coder`` is a
# strict superset for value coercion — it accepts Python literals via
# ``ast.literal_eval`` where ``qwen3_xml`` returns raw strings. Since
# this repo doesn't stream yet, ``qwen3_coder`` semantics dominate.
# TODO: when a streaming parse API lands, switch the streaming path to
# ``qwen3_xml``-style state-machine semantics to dodge vLLM's
# regex-streaming bugs (see HF Qwen3-Coder-Next discussion #17).


_STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char", "enum"})
_BOOL_TYPES = frozenset({"boolean", "bool", "binary"})
_OBJECT_TYPES = frozenset({"object", "array", "arr"})
_INT_PREFIXES = ("int", "uint", "long", "short", "unsigned")
_FLOAT_PREFIXES = ("num", "float")
_OBJECT_PREFIXES = ("dict", "list")


def _build_param_type_index(
    tools: list[ToolSpec] | None,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Map tool name → param name → param JSON-schema fragment.

    Accepts both flat ``ToolSpec`` (``{name, description, parameters}``)
    and the OpenAI envelope (``{"type": "function", "function": {...}}``)
    so callers can pass either shape.
    """
    if not tools:
        return {}
    index: dict[str, dict[str, dict[str, Any]]] = {}
    for tool in tools:
        spec = tool.get("function", tool) if isinstance(tool, dict) else None
        if not isinstance(spec, dict):
            continue
        name = spec.get("name")
        if not isinstance(name, str):
            continue
        params = spec.get("parameters") or {}
        props = params.get("properties") if isinstance(params, dict) else None
        if isinstance(props, dict):
            index[name] = {k: v for k, v in props.items() if isinstance(v, dict)}
    return index


def _declared_type(param_schema: dict[str, Any] | None) -> str | None:
    """Extract the lowercased ``type`` from a JSON-schema fragment.

    Mirrors vLLM: a fragment with ``anyOf`` and no top-level ``type`` is
    treated as ``"object"`` so json.loads is attempted; a list-form type
    falls back to the first declared entry.
    """
    if not isinstance(param_schema, dict):
        return None
    declared = param_schema.get("type")
    if isinstance(declared, str):
        return declared.strip().lower()
    if isinstance(declared, list) and declared:
        first = declared[0]
        if isinstance(first, str):
            return first.strip().lower()
    if "anyOf" in param_schema:
        return "object"
    return None


def _coerce_arg_value(
    text: str, param_schema: dict[str, Any] | None
) -> tuple[Any, bool]:
    """Coerce a raw ``<parameter=…>`` body to its declared type.

    Returns ``(value, used_fallback)``. ``used_fallback=True`` means the
    coercion could not satisfy the declared type and had to fall back to
    a different shape (a string for scalars, ``False`` for booleans, raw
    text for objects), so the caller can flag ``INVALID_JSON``.

    The ladder mirrors vLLM's ``Qwen3CoderToolParser._convert_param_value``
    and SGLang's ``Qwen3CoderDetector._convert_param_value`` with one
    deliberate deviation noted inline:

    - ``null`` short-circuit (any case) → ``None`` for any declared
      type **except** the string family (see deviation below).
    - No schema OR param not in schema → return verbatim (after the
      null short-circuit). vLLM ``qwen3coder_tool_parser.py:128-137``:
      ``if param_name not in param_config: return param_value``.
    - ``string`` / ``str`` / ``text`` / ``varchar`` / ``char`` / ``enum``
      → return verbatim. (Deviation: vLLM/SGLang null-coerce ``"null"``
      *before* checking type, so a string-typed arg of ``"null"`` would
      come back as Python ``None``. Renderers preserves the string
      because the XML wire format can't distinguish the string
      ``"null"`` from JSON null, and downstream tests pin the
      preservation contract; see ``test_tool_arg_type_preservation``.)
    - ``int`` / ``uint`` / ``long`` / ``short`` / ``unsigned`` family →
      ``int(text)``, degenerating to raw text + INVALID.
    - ``num`` / ``float`` family → ``float(text)`` with int demotion
      when the source has no ``.``/``e`` and is whole, degenerating to
      raw text + INVALID.
    - ``boolean`` / ``bool`` / ``binary`` → ``text.lower() == "true"``;
      anything outside ``{"true", "false"}`` returns ``False`` + INVALID
      (matches vLLM's "degenerate to false" behavior, with the
      INVALID flag preserved for the verifier / RL-loss signal).
    - ``object`` / ``array`` / ``dict`` / ``list`` (or ``anyOf``) →
      ``json.loads`` first, ``ast.literal_eval`` fallback (for Python
      literals like ``{'k': 1}``), raw text + INVALID otherwise.
    - Any other declared type → ``ast.literal_eval`` if it parses, else
      raw text. Matches vLLM's catch-all else branch.
    """
    declared = _declared_type(param_schema)

    # vLLM null short-circuit runs BEFORE the schema-missing check, but
    # is suppressed for declared string types so the wire-format
    # ambiguity is resolved in favour of the explicit schema.
    if declared not in _STRING_TYPES and text.lower() == "null":
        return None, False

    if param_schema is None or declared is None or declared in _STRING_TYPES:
        return text, False

    if declared.startswith(_INT_PREFIXES):
        try:
            return int(text), False
        except (ValueError, TypeError):
            return text, True

    if declared.startswith(_FLOAT_PREFIXES):
        try:
            f = float(text)
        except (ValueError, TypeError):
            return text, True
        if "." not in text and "e" not in text.lower() and f.is_integer():
            return int(f), False
        return f, False

    if declared in _BOOL_TYPES:
        lowered = text.lower()
        if lowered == "true":
            return True, False
        if lowered == "false":
            return False, False
        return False, True

    if declared in _OBJECT_TYPES or declared.startswith(_OBJECT_PREFIXES):
        try:
            return json.loads(text), False
        except (json.JSONDecodeError, ValueError):
            pass
        try:
            return ast.literal_eval(text), False
        except (ValueError, SyntaxError, TypeError, MemoryError):
            return text, True

    try:
        return ast.literal_eval(text), False
    except (ValueError, SyntaxError, TypeError, MemoryError):
        return text, True


def _find(ids: list[int], target: int, start: int = 0) -> int:
    """Find index of target in ids, or -1."""
    for i in range(start, len(ids)):
        if ids[i] == target:
            return i
    return -1


def _find_any(ids: list[int], targets: set[int], start: int = 0) -> int:
    """Find first index in ids whose value is in targets, or -1."""
    for i in range(start, len(ids)):
        if ids[i] in targets:
            return i
    return -1


def _find_all(ids: list[int], target: int) -> list[int]:
    """Find all indices of target in ids."""
    return [i for i, t in enumerate(ids) if t == target]


def _strip_stop_tokens(ids: list[int], stop_ids: set[int]) -> list[int]:
    """Truncate at first stop token (model shouldn't generate past it)."""
    for i, t in enumerate(ids):
        if t in stop_ids:
            return ids[:i]
    return ids


def _decode(tokenizer, ids: list[int]) -> str:
    """Decode token IDs to text, skipping special tokens."""
    if not ids:
        return ""
    return tokenizer.decode(ids, skip_special_tokens=False)


# ── Qwen3: <tool_call> JSON </tool_call> ────────────────────────────


def parse_qwen3(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_call_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Qwen3 completion tokens. Hermes-style JSON tool calls."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_ids = ids[:tc_start]
        i = tc_start
        while i < len(ids):
            if ids[i] == tool_call_id:
                end = _find(ids, tool_call_end_id, i + 1)
                if end == -1:
                    # No closing delim — block runs to end of stripped ids.
                    raw = _decode(tokenizer, ids[i + 1 :]).strip()
                    tool_calls.append(
                        ParsedToolCall(
                            raw=raw,
                            token_span=(i, len(ids)),
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                tc_text = _decode(tokenizer, ids[i + 1 : end]).strip()
                span = (i, end + 1)
                try:
                    parsed = json.loads(tc_text)
                except json.JSONDecodeError:
                    tool_calls.append(
                        ParsedToolCall(
                            raw=tc_text,
                            token_span=span,
                            status=ToolCallParseStatus.INVALID_JSON,
                        )
                    )
                else:
                    name = parsed.get("name", "") if isinstance(parsed, dict) else ""
                    arguments = (
                        parsed.get("arguments", {}) if isinstance(parsed, dict) else {}
                    )
                    if not name:
                        tool_calls.append(
                            ParsedToolCall(
                                raw=tc_text,
                                name=None,
                                arguments=arguments,
                                token_span=span,
                                status=ToolCallParseStatus.MISSING_NAME,
                            )
                        )
                    else:
                        tool_calls.append(
                            ParsedToolCall(
                                raw=tc_text,
                                name=name,
                                arguments=arguments,
                                token_span=span,
                                status=ToolCallParseStatus.OK,
                            )
                        )
                i = end + 1
            else:
                i += 1
    else:
        content_ids = ids

    text = _decode(tokenizer, content_ids)
    # Extract reasoning from text (Qwen3 doesn't have <think> as special token)
    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").strip("\n").strip()
        text = after.strip("\n")

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


# ── Qwen3.5: <tool_call> <function=name> <parameter=name> v </parameter> </function> </tool_call>


def parse_qwen35(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse Qwen3.5 completion tokens. XML-style tool calls, token-level thinking."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # Thinking: find </think> by token ID
    reasoning = None
    parse_offset = 0  # shift to map local indices back to stop-stripped ids
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif think_id in set(ids):
        # <think> present but no </think> — truncated reasoning
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    full_text = _decode(tokenizer, ids)
    tc_start = _find(ids, tool_call_id)
    param_index = _build_param_type_index(tools)
    try:
        if tc_start != -1:
            # vLLM ``qwen3coder_tool_parser.py:316-319``: content text
            # is the raw slice up to the first ``<tool_call>`` (or
            # first ``<function=`` if no ``<tool_call>`` is present).
            # No whitespace stripping — the model's prefix prose is
            # surfaced verbatim.
            content_text = _decode(tokenizer, ids[:tc_start])
            tool_calls = _parse_xml_tool_calls(
                tokenizer,
                ids[tc_start:],
                tool_call_id,
                tool_call_end_id,
                section_offset=parse_offset + tc_start,
                param_index=param_index,
            )
        else:
            # vLLM ``qwen3coder_tool_parser.py:269-271`` back-off: no
            # ``<tool_call>`` markers — scan whole output for raw
            # ``<function=…>`` blocks. Content text is whatever sits
            # before the first ``<function=`` (vLLM line 317-319), or
            # the full text if nothing matches.
            tool_calls = _parse_xml_function_blocks(
                full_text,
                param_index=param_index,
                token_span=(parse_offset, parse_offset + len(ids)),
                wrapper_unclosed=False,
            )
            if tool_calls:
                first_fn = full_text.find("<function=")
                content_text = full_text[:first_fn] if first_fn >= 0 else full_text
            else:
                content_text = full_text
    except Exception:
        # vLLM ``qwen3coder_tool_parser.py:327-331`` catch-all: on any
        # extraction error, drop every recovered call and surface the
        # raw text as content so downstream consumers never see a
        # half-parsed call.
        content_text = full_text
        tool_calls = []

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_xml_tool_calls(
    tokenizer,
    ids: list[int],
    tc_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
    param_index: dict[str, dict[str, dict[str, Any]]],
) -> list[ParsedToolCall]:
    """Parse Qwen3.5-style XML tool calls from token IDs.

    Uses token IDs to demarcate ``<tool_call>`` / ``</tool_call>``
    boundaries (so ``token_span`` stays precise for trainer-side
    masking) and vLLM-parity regex on the decoded block text to
    extract the function/parameter content tolerantly. Mirrors
    ``Qwen3CoderToolParser._parse_xml_function_call`` plus the
    tag-tolerance branches of the three module-level patterns.
    """
    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] != tc_id:
            i += 1
            continue

        end = _find(ids, tc_end_id, i + 1)
        if end == -1:
            block_ids = ids[i + 1 :]
            block_text = _decode(tokenizer, block_ids)
            span = (section_offset + i, section_offset + len(ids))
            wrapper_unclosed = True
        else:
            block_text = _decode(tokenizer, ids[i + 1 : end])
            span = (section_offset + i, section_offset + end + 1)
            wrapper_unclosed = False

        block_calls = _parse_xml_function_blocks(
            block_text,
            param_index=param_index,
            token_span=span,
            wrapper_unclosed=wrapper_unclosed,
        )
        if block_calls:
            tool_calls.extend(block_calls)
        elif wrapper_unclosed:
            # vLLM would silently drop a content-less unclosed
            # ``<tool_call>``; we keep the diagnostic so verifier /
            # RL-loss code can mask the malformed span.
            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    token_span=span,
                    status=ToolCallParseStatus.UNCLOSED_BLOCK,
                )
            )
        else:
            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    token_span=span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            )

        if wrapper_unclosed:
            break
        i = end + 1
    return tool_calls


def _parse_xml_function_blocks(
    text: str,
    *,
    param_index: dict[str, dict[str, dict[str, Any]]],
    token_span: tuple[int, int] | None,
    wrapper_unclosed: bool,
) -> list[ParsedToolCall]:
    """Apply vLLM's ``<function=…></function>`` regex over *text*.

    ``token_span`` is shared by every call recovered from this block;
    the granularity is the surrounding ``<tool_call>`` region (or the
    whole completion in the no-marker back-off path), matching how
    vLLM treats multiple ``<function=>`` siblings — it does not try to
    attribute character offsets back to token positions.
    """
    tool_calls: list[ParsedToolCall] = []
    for match in _FUNCTION_BLOCK_RE.finditer(text):
        closed = match.group(1)
        body = closed if closed is not None else (match.group(2) or "")
        function_unclosed = closed is None
        end_index = body.find(">")
        if end_index == -1:
            tool_calls.append(
                ParsedToolCall(
                    raw=body,
                    token_span=token_span,
                    status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                )
            )
            continue

        name = body[:end_index]
        params_text = body[end_index + 1 :]
        params = param_index.get(name, {})
        arguments: dict = {}
        any_fallback = False
        for capture in _PARAMETER_BLOCK_RE.findall(params_text):
            idx = capture.find(">")
            if idx == -1:
                continue
            arg_name = capture[:idx]
            arg_value = _strip_one_newline(capture[idx + 1 :])
            value, used_fallback = _coerce_arg_value(arg_value, params.get(arg_name))
            arguments[arg_name] = value
            any_fallback = any_fallback or used_fallback

        if wrapper_unclosed or function_unclosed:
            status = ToolCallParseStatus.UNCLOSED_BLOCK
        elif any_fallback:
            status = ToolCallParseStatus.INVALID_JSON
        else:
            status = ToolCallParseStatus.OK
        tool_calls.append(
            ParsedToolCall(
                raw=body,
                name=name,
                arguments=arguments,
                id=_make_tool_call_id(),
                token_span=token_span,
                status=status,
            )
        )
    return tool_calls


def _strip_one_newline(s: str) -> str:
    """vLLM ``qwen3coder_tool_parser.py:246-250`` strips exactly one
    leading and one trailing newline from a parameter value — no
    further whitespace trimming."""
    if s.startswith("\n"):
        s = s[1:]
    if s.endswith("\n"):
        s = s[:-1]
    return s


# ── GLM-5/4.7/4.5: <tool_call> name <arg_key>k</arg_key> <arg_value>v</arg_value> </tool_call>


def parse_glm(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    arg_key_id: int,
    arg_key_end_id: int,
    arg_value_id: int,
    arg_value_end_id: int,
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse GLM completion tokens. Token-level thinking + arg_key/arg_value tool calls."""
    ids = _strip_stop_tokens(token_ids, stop_ids)

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif think_id in set(ids):
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        tool_calls = _parse_glm_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            arg_key_id,
            arg_key_end_id,
            arg_value_id,
            arg_value_end_id,
            section_offset=parse_offset + tc_start,
            param_index=_build_param_type_index(tools),
        )
    else:
        content_text = _decode(tokenizer, ids).strip()

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_glm_tool_calls(
    tokenizer,
    ids,
    tc_id,
    tc_end_id,
    ak_id,
    ake_id,
    av_id,
    ave_id,
    *,
    section_offset: int,
    param_index: dict[str, dict[str, dict[str, Any]]],
) -> list[ParsedToolCall]:
    """Parse GLM-style tool calls: name + arg_key/arg_value pairs, all by token ID."""
    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_id:
            end = _find(ids, tc_end_id, i + 1)
            if end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            block = ids[i + 1 : end]
            block_text = _decode(tokenizer, block)
            span = (section_offset + i, section_offset + end + 1)
            first_ak = _find(block, ak_id)
            any_json_fallback = False
            structure_broke = False
            if first_ak == -1:
                name = _decode(tokenizer, block).strip()
                arguments: dict = {}
            else:
                name = _decode(tokenizer, block[:first_ak]).strip()
                params = param_index.get(name, {})
                arguments = {}
                j = first_ak
                while j < len(block):
                    if block[j] == ak_id:
                        ake = _find(block, ake_id, j + 1)
                        if ake == -1:
                            structure_broke = True
                            break
                        key = _decode(tokenizer, block[j + 1 : ake]).strip()
                        av = _find(block, av_id, ake + 1)
                        if av == -1:
                            structure_broke = True
                            break
                        ave = _find(block, ave_id, av + 1)
                        if ave == -1:
                            structure_broke = True
                            break
                        val_text = _decode(tokenizer, block[av + 1 : ave]).strip()
                        value, used_fallback = _coerce_arg_value(
                            val_text, params.get(key)
                        )
                        arguments[key] = value
                        any_json_fallback = any_json_fallback or used_fallback
                        j = ave + 1
                    else:
                        j += 1
            if not name:
                status = ToolCallParseStatus.MISSING_NAME
            elif structure_broke:
                status = ToolCallParseStatus.MALFORMED_STRUCTURE
            elif any_json_fallback:
                status = ToolCallParseStatus.INVALID_JSON
            else:
                status = ToolCallParseStatus.OK
            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    name=name or None,
                    arguments=arguments,
                    token_span=span,
                    status=status,
                )
            )
            i = end + 1
        else:
            i += 1
    return tool_calls


# ── Laguna-XS.2: <tool_call> name\n<arg_key>k</arg_key>\n<arg_value>v</arg_value> </tool_call>
# Same outer skeleton as parse_glm, but <arg_key>/<arg_value> are plain text
# (multi-token BPE), not single special tokens — so the inner block is decoded
# to text and the key/value pairs are pulled out by regex.


def parse_laguna_xs2(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse Laguna-XS.2 completion tokens.

    Thinking uses single-token ``<think>`` / ``</think>`` (ids found by
    scan). Tool calls are delimited by single-token ``<tool_call>`` /
    ``</tool_call>``, but ``<arg_key>`` / ``<arg_value>`` inside are
    plain text — regex-extracted from the decoded inner block.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    # The template wraps reasoning with ``\n`` on both sides
    # (``<think>\n{r}\n</think>``) and brackets post-think content with ``\n``
    # too (``</think>\n{c}\n``). Strip exactly those newlines from each
    # decoded segment — never a bare ``.strip()``, which would also eat
    # whitespace the model emitted intentionally.
    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip("\n")
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif (think_start := _find(ids, think_id)) != -1:
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip("\n")
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip("\n")
        tool_calls = _parse_laguna_xs2_tool_calls(
            tokenizer,
            ids[tc_start:],
            tool_call_id,
            tool_call_end_id,
            section_offset=parse_offset + tc_start,
            param_index=_build_param_type_index(tools),
        )
    else:
        content_text = _decode(tokenizer, ids).strip("\n")

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_laguna_xs2_tool_calls(
    tokenizer,
    ids: list[int],
    tc_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
    param_index: dict[str, dict[str, dict[str, Any]]],
) -> list[ParsedToolCall]:
    """Parse Laguna-XS.2 tool calls.

    Inside each ``<tool_call>...</tool_call>`` block, the format is::

        {name}\\n
        <arg_key>{k1}</arg_key>\\n<arg_value>{v1}</arg_value>\\n
        ...
        <arg_key>{kn}</arg_key>\\n<arg_value>{vn}</arg_value>\\n

    The function name is everything before the first ``<arg_key>`` literal
    in the decoded block.
    """
    import re

    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_id:
            tc_end = _find(ids, tc_end_id, i + 1)
            if tc_end == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.UNCLOSED_BLOCK,
                    )
                )
                break
            block_text = _decode(tokenizer, ids[i + 1 : tc_end])
            span = (section_offset + i, section_offset + tc_end + 1)

            ak_pos = block_text.find("<arg_key>")
            if ak_pos != -1:
                name = block_text[:ak_pos].strip()
                args_section = block_text[ak_pos:]
            else:
                name = block_text.strip()
                args_section = ""

            params = param_index.get(name, {})
            arguments: dict = {}
            any_json_fallback = False
            for m in re.finditer(
                r"<arg_key>(.*?)</arg_key>\s*<arg_value>(.*?)</arg_value>",
                args_section,
                re.DOTALL,
            ):
                k = m.group(1).strip()
                v = m.group(2).strip()
                value, used_fallback = _coerce_arg_value(v, params.get(k))
                arguments[k] = value
                any_json_fallback = any_json_fallback or used_fallback

            if not name:
                status = ToolCallParseStatus.MISSING_NAME
            elif any_json_fallback:
                status = ToolCallParseStatus.INVALID_JSON
            else:
                status = ToolCallParseStatus.OK

            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    name=name or None,
                    arguments=arguments,
                    token_span=span,
                    status=status,
                )
            )
            i = tc_end + 1
        else:
            i += 1
    return tool_calls


# ── DeepSeek V3: <｜tool▁calls▁begin｜>...<｜tool▁calls▁end｜> + text <think> tags ──


def parse_deepseek_v3(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_calls_begin_id: int,
    tool_calls_end_id: int,
    tool_call_begin_id: int,
    tool_call_end_id: int,
    tool_sep_id: int,
) -> ParsedResponse:
    """Parse DeepSeek V3 completion tokens.

    Thinking is embedded as plain text <think>...</think> tags (not special tokens).
    Tool calls are delimited by special tokens:
        <｜tool▁calls▁begin｜>
          <｜tool▁call▁begin｜>function<｜tool▁sep｜>{name}\\n```json\\n{args}\\n```<｜tool▁call▁end｜>
        <｜tool▁calls▁end｜>
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    tc_section_start = _find(ids, tool_calls_begin_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_section_start != -1:
        content_ids = ids[:tc_section_start]
        tool_calls = _parse_deepseek_tool_calls(
            tokenizer,
            ids[tc_section_start:],
            tool_calls_begin_id,
            tool_calls_end_id,
            tool_call_begin_id,
            tool_call_end_id,
            tool_sep_id,
            section_offset=tc_section_start,
        )
    else:
        content_ids = ids

    text = _decode(tokenizer, content_ids)

    reasoning = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        reasoning = before.replace("<think>", "").lstrip("\n").rstrip("\n").strip()
        text = after.lstrip("\n")

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


def _parse_deepseek_tool_calls(
    tokenizer,
    ids: list[int],
    tc_begin_id: int,
    tc_end_id: int,
    call_begin_id: int,
    call_end_id: int,
    sep_id: int,
    *,
    section_offset: int,
) -> list[ParsedToolCall]:
    """Parse DeepSeek V3-style tool calls from token IDs."""
    import re

    tool_calls: list[ParsedToolCall] = []

    section_start = _find(ids, tc_begin_id)
    if section_start == -1:
        return tool_calls
    section_end = _find(ids, tc_end_id, section_start + 1)
    section_end_clipped = section_end == -1
    if section_end == -1:
        section_end = len(ids)

    inner_offset = section_offset + section_start + 1
    section_ids = ids[section_start + 1 : section_end]

    i = 0
    while i < len(section_ids):
        if section_ids[i] == call_begin_id:
            end = _find(section_ids, call_end_id, i + 1)
            unclosed = end == -1
            if unclosed:
                end = len(section_ids)
            call_ids = section_ids[i + 1 : end]
            block_text = _decode(tokenizer, call_ids)
            # Span for this call covers its <tool_call_begin>..<tool_call_end> range
            # within the (stop-stripped) parent token stream.
            span = (
                inner_offset + i,
                inner_offset + end + (0 if unclosed else 1),
            )

            sep_pos = _find(call_ids, sep_id)
            if sep_pos == -1:
                tool_calls.append(
                    ParsedToolCall(
                        raw=block_text,
                        token_span=span,
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                i = end + 1
                continue

            after_sep_ids = call_ids[sep_pos + 1 :]
            after_sep_text = _decode(tokenizer, after_sep_ids).strip()

            name = ""
            args_str = ""
            newline_pos = after_sep_text.find("\n")
            if newline_pos != -1:
                name = after_sep_text[:newline_pos].strip()
                rest = after_sep_text[newline_pos + 1 :].strip()
                fence_match = re.match(r"```(?:json)?\s*([\s\S]*?)\s*```$", rest)
                args_str = fence_match.group(1).strip() if fence_match else rest
            else:
                name = after_sep_text

            arguments: dict | str
            invalid_json = False
            try:
                arguments = json.loads(args_str) if args_str else {}
            except json.JSONDecodeError:
                arguments = args_str
                invalid_json = True

            if unclosed:
                status = ToolCallParseStatus.UNCLOSED_BLOCK
            elif not name:
                status = ToolCallParseStatus.MISSING_NAME
            elif invalid_json:
                status = ToolCallParseStatus.INVALID_JSON
            else:
                status = ToolCallParseStatus.OK

            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    name=name or None,
                    arguments=arguments,
                    token_span=span,
                    status=status,
                )
            )
            i = end + 1
            if unclosed:
                break
        else:
            i += 1

    # If the outer <tool_calls_begin> had no matching <tool_calls_end>, any
    # call inside that didn't itself flag UNCLOSED_BLOCK is still nested in
    # a truncated section — but we already mark individual unclosed calls,
    # so we don't double-flag here. The section_end_clipped variable is
    # carried for the (rare) caller that wants section-level UX.
    _ = section_end_clipped
    return tool_calls


# ── MiniMax: <minimax:tool_call> ... </minimax:tool_call> ────────────


def parse_minimax(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    think_id: int,
    think_end_id: int,
    tool_call_id: int,
    tool_call_end_id: int,
    tools: list[ToolSpec] | None = None,
) -> ParsedResponse:
    """Parse MiniMax M2 completion tokens."""
    import re

    ids = _strip_stop_tokens(token_ids, stop_ids)
    param_index = _build_param_type_index(tools)

    reasoning = None
    parse_offset = 0
    think_end = _find(ids, think_end_id)
    if think_end != -1:
        reasoning_ids = ids[:think_end]
        reasoning_ids = [t for t in reasoning_ids if t != think_id]
        reasoning = _decode(tokenizer, reasoning_ids).strip()
        ids = ids[think_end + 1 :]
        parse_offset = think_end + 1
    elif think_id in set(ids):
        think_start = _find(ids, think_id)
        reasoning = _decode(tokenizer, ids[think_start + 1 :]).strip()
        return ParsedResponse(
            content="", reasoning_content=reasoning or None, tool_calls=[]
        )

    tc_start = _find(ids, tool_call_id)
    tool_calls: list[ParsedToolCall] = []
    if tc_start != -1:
        content_text = _decode(tokenizer, ids[:tc_start]).strip()
        i = tc_start
        while i < len(ids):
            if ids[i] == tool_call_id:
                end = _find(ids, tool_call_end_id, i + 1)
                if end == -1:
                    raw = _decode(tokenizer, ids[i + 1 :])
                    tool_calls.append(
                        ParsedToolCall(
                            raw=raw,
                            token_span=(
                                parse_offset + i,
                                parse_offset + len(ids),
                            ),
                            status=ToolCallParseStatus.UNCLOSED_BLOCK,
                        )
                    )
                    break
                block_text = _decode(tokenizer, ids[i + 1 : end])
                span = (parse_offset + i, parse_offset + end + 1)

                invokes = list(
                    re.finditer(
                        r'<invoke name="([^"]+)">(.*?)</invoke>',
                        block_text,
                        re.DOTALL,
                    )
                )
                if not invokes:
                    # Block exists but contains no <invoke> — model emitted
                    # the wrapper without a usable body.
                    tool_calls.append(
                        ParsedToolCall(
                            raw=block_text,
                            token_span=span,
                            status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                        )
                    )
                else:
                    for invoke_match in invokes:
                        name = invoke_match.group(1)
                        body = invoke_match.group(2)
                        params = param_index.get(name, {})
                        arguments: dict = {}
                        any_json_fallback = False
                        for pm in re.finditer(
                            r'<parameter name="([^"]+)">(.*?)</parameter>',
                            body,
                            re.DOTALL,
                        ):
                            pname = pm.group(1)
                            pval = pm.group(2).strip()
                            value, used_fallback = _coerce_arg_value(
                                pval, params.get(pname)
                            )
                            arguments[pname] = value
                            any_json_fallback = any_json_fallback or used_fallback
                        tool_calls.append(
                            ParsedToolCall(
                                raw=block_text,
                                name=name,
                                arguments=arguments,
                                # All invokes in a block share the wrapper span.
                                token_span=span,
                                status=(
                                    ToolCallParseStatus.INVALID_JSON
                                    if any_json_fallback
                                    else ToolCallParseStatus.OK
                                ),
                            )
                        )
                i = end + 1
            else:
                i += 1
    else:
        content_text = _decode(tokenizer, ids).strip()

    return ParsedResponse(
        content=content_text,
        reasoning_content=reasoning or None,
        tool_calls=tool_calls,
    )


# ── Kimi K2: <|tool_calls_section_begin|> ... <|tool_calls_section_end|> ────


def parse_kimi_k2_section(
    tokenizer,
    ids: list[int],
    *,
    tool_calls_section_begin_ids: set[int],
    tool_calls_section_end_ids: set[int],
    tool_call_begin_id: int,
    tool_call_argument_begin_id: int,
    tool_call_end_id: int,
) -> tuple[list[int], list[ParsedToolCall]]:
    """Split ``ids`` into ``(content_before_section, tool_calls)`` by finding
    the Kimi-style tool-call section delimiters.

    Accepts *sets* of begin/end token IDs so callers can express models with
    multiple delimiter variants (K2.5 has both plural ``<|tool_calls_section_*|>``
    and singular ``<|tool_call_section_*|>`` forms, though only the plural form
    is in the special-token vocab in practice). Returns the content ids ahead
    of the section and a list of ``ParsedToolCall`` covering every attempted
    block inside it; an unclosed section is still walked to whatever the model
    emitted before EOS. Returns ``(ids, [])`` when no section is present.
    """
    section_start = _find_any(ids, tool_calls_section_begin_ids)
    if section_start == -1:
        return list(ids), []
    content_ids = ids[:section_start]
    section_end = _find_any(ids, tool_calls_section_end_ids, section_start + 1)
    if section_end == -1:
        section_end = len(ids)
    section_ids = ids[section_start + 1 : section_end]
    tool_calls = _parse_kimi_k2_tool_calls(
        tokenizer,
        section_ids,
        tool_call_begin_id,
        tool_call_argument_begin_id,
        tool_call_end_id,
        section_offset=section_start + 1,
    )
    return content_ids, tool_calls


def parse_kimi_k2(
    tokenizer,
    token_ids: list[int],
    *,
    stop_ids: set[int],
    tool_calls_section_begin_id: int,
    tool_calls_section_end_id: int,
    tool_call_begin_id: int,
    tool_call_argument_begin_id: int,
    tool_call_end_id: int,
) -> ParsedResponse:
    """Parse Kimi K2 completion tokens.

    Thinking is encoded as text tags <think>...</think>.
    Tool calls use section/call-level special tokens.
    Tool call IDs are in format ``functions.name:index``.
    """
    ids = _strip_stop_tokens(token_ids, stop_ids)

    content_ids, tool_calls = parse_kimi_k2_section(
        tokenizer,
        ids,
        tool_calls_section_begin_ids={tool_calls_section_begin_id},
        tool_calls_section_end_ids={tool_calls_section_end_id},
        tool_call_begin_id=tool_call_begin_id,
        tool_call_argument_begin_id=tool_call_argument_begin_id,
        tool_call_end_id=tool_call_end_id,
    )

    text = _decode(tokenizer, content_ids)
    reasoning: str | None = None
    if "</think>" in text:
        before, _, after = text.partition("</think>")
        raw_think = before.replace("<think>", "", 1)
        reasoning = raw_think.strip("\n").strip() or None
        text = after.strip("\n")
    elif "<think>" in text:
        # Truncated thinking (no closing tag)
        raw_think = text.split("<think>", 1)[1]
        reasoning = raw_think.strip("\n").strip() or None
        return ParsedResponse(
            content="",
            reasoning_content=reasoning,
            tool_calls=[],
        )

    return ParsedResponse(
        content=text.strip(),
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )


def _parse_kimi_k2_tool_calls(
    tokenizer,
    ids: list[int],
    tc_begin_id: int,
    tc_arg_begin_id: int,
    tc_end_id: int,
    *,
    section_offset: int,
) -> list[ParsedToolCall]:
    """Parse individual Kimi K2 tool calls from the section token IDs.

    Format per call:
        <|tool_call_begin|>{id}<|tool_call_argument_begin|>{json_args}<|tool_call_end|>

    The ``id`` is in format ``functions.name:index``; the function name is
    extracted by stripping the ``functions.`` prefix and ``:index`` suffix.
    """
    tool_calls: list[ParsedToolCall] = []
    i = 0
    while i < len(ids):
        if ids[i] == tc_begin_id:
            arg_begin = _find(ids, tc_arg_begin_id, i + 1)
            if arg_begin == -1:
                raw = _decode(tokenizer, ids[i + 1 :])
                tool_calls.append(
                    ParsedToolCall(
                        raw=raw,
                        token_span=(section_offset + i, section_offset + len(ids)),
                        status=ToolCallParseStatus.MALFORMED_STRUCTURE,
                    )
                )
                break
            tc_end = _find(ids, tc_end_id, arg_begin + 1)
            unclosed = tc_end == -1
            if tc_end == -1:
                tc_end = len(ids)

            raw_id = _decode(tokenizer, ids[i + 1 : arg_begin]).strip()
            args_str = _decode(tokenizer, ids[arg_begin + 1 : tc_end]).strip()
            block_text = _decode(tokenizer, ids[i + 1 : tc_end])
            span = (
                section_offset + i,
                section_offset + tc_end + (0 if unclosed else 1),
            )

            name_part = raw_id.split(":", 1)[0]
            if "." in name_part:
                _, func_name = name_part.split(".", 1)
            else:
                func_name = name_part

            arguments: dict | str
            invalid_json = False
            try:
                arguments = json.loads(args_str)
            except json.JSONDecodeError:
                arguments = args_str
                invalid_json = True

            if unclosed:
                status = ToolCallParseStatus.UNCLOSED_BLOCK
            elif not func_name:
                status = ToolCallParseStatus.MISSING_NAME
            elif invalid_json:
                status = ToolCallParseStatus.INVALID_JSON
            else:
                status = ToolCallParseStatus.OK

            tool_calls.append(
                ParsedToolCall(
                    raw=block_text,
                    name=func_name or None,
                    arguments=arguments,
                    token_span=span,
                    status=status,
                    id=raw_id or None,
                )
            )
            i = tc_end + 1
            if unclosed:
                break
        else:
            i += 1
    return tool_calls


# ── GptOss (Harmony): <|start|>role<|channel|>ch<|message|>content<|end|/return|/call|>


def parse_gpt_oss(
    tokenizer,
    token_ids: list[int],
    *,
    return_id: int,
    call_id: int,
    start_id: int,
    end_id: int,
    channel_id: int,
    message_id: int,
    constrain_id: int,
) -> ParsedResponse:
    """Parse GptOss (Harmony) completion tokens.

    Finds the earliest terminal token (<|return|> or <|call|>), then walks the
    token stream block-by-block to extract:

    - analysis channel              → reasoning_content
    - final channel                 → content
    - commentary with to=functions.*  → tool_calls (JSON arguments)
    - commentary without recipient  → content (preamble text)
    """
    import re

    # Only <|return|> terminates the whole turn. <|call|> closes an
    # individual tool-call commentary block — a single turn may contain
    # several, so we must NOT truncate at the first <|call|>.
    return_pos = _find(token_ids, return_id)
    if return_pos != -1:
        ids = token_ids[:return_pos]
    else:
        ids = list(token_ids)

    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[ParsedToolCall] = []

    i = 0
    while i < len(ids):
        if ids[i] != start_id:
            i += 1
            continue

        block_start = i
        msg_pos = _find(ids, message_id, i + 1)
        if msg_pos == -1:
            break

        header_ids = ids[i + 1 : msg_pos]
        header_text = _decode(tokenizer, header_ids)

        body_start = msg_pos + 1
        candidates = [
            pos
            for pos in (
                _find(ids, start_id, body_start),
                _find(ids, end_id, body_start),
                _find(ids, call_id, body_start),
            )
            if pos != -1
        ]
        body_end = min(candidates) if candidates else len(ids)
        body_closed = bool(candidates) and ids[body_end] in (end_id, call_id)

        body_text = _decode(tokenizer, ids[body_start:body_end])

        channel = _gptoss_extract_after_token(tokenizer, header_ids, channel_id)

        recipient_match = re.search(r"to=([^\s<]+)", header_text)
        recipient = recipient_match.group(1) if recipient_match else None

        if recipient and recipient.startswith("functions."):
            tool_name = recipient[len("functions.") :]
            block_end = body_end + 1 if body_closed else body_end
            span = (block_start, block_end)
            try:
                arguments = json.loads(body_text)
            except json.JSONDecodeError:
                tool_calls.append(
                    ParsedToolCall(
                        raw=body_text,
                        name=tool_name or None,
                        arguments=body_text,
                        token_span=span,
                        status=ToolCallParseStatus.INVALID_JSON,
                    )
                )
            else:
                if not body_closed:
                    status = ToolCallParseStatus.UNCLOSED_BLOCK
                elif not tool_name:
                    status = ToolCallParseStatus.MISSING_NAME
                else:
                    status = ToolCallParseStatus.OK
                tool_calls.append(
                    ParsedToolCall(
                        raw=body_text,
                        name=tool_name or None,
                        arguments=arguments,
                        token_span=span,
                        status=status,
                    )
                )
        elif channel == "analysis":
            reasoning_parts.append(body_text)
        elif channel == "final":
            content_parts.append(body_text)
        elif channel == "commentary":
            content_parts.append(body_text)

        i = body_end
        if i < len(ids) and ids[i] in (end_id, call_id):
            i += 1

    reasoning = "".join(reasoning_parts).strip() or None
    content = "".join(content_parts).strip()

    return ParsedResponse(
        content=content,
        reasoning_content=reasoning,
        tool_calls=tool_calls,
    )


def _gptoss_extract_after_token(
    tokenizer,
    header_ids: list[int],
    marker_id: int,
) -> str | None:
    """Return the first decoded word appearing after marker_id in header_ids."""
    pos = _find(header_ids, marker_id)
    if pos == -1:
        return None
    after = _decode(tokenizer, header_ids[pos + 1 :]).strip()
    return after.split()[0] if after else None
