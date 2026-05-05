"""Smoke coverage for the ``preserve_*_thinking`` override flags.

Two invariants per renderer:

1. Default render (both flags ``False``) is byte-identical to the existing
   ``apply_chat_template`` parity baseline — covered exhaustively elsewhere.
2. Setting either flag never *removes* tokens compared to the default and,
   for renderers whose template would drop past-asst thinking, actually
   adds tokens for a conversation containing past-asst ``reasoning_content``.

Renderers whose template either always preserves thinking (DeepSeek-V3) or
never references ``reasoning_content`` for past-asst (Kimi-K2, Qwen3-VL)
are no-ops by design — they're listed below and the test asserts the
default==override equality instead of strict growth.
"""

from __future__ import annotations

import pytest

from renderers.base import should_preserve_past_thinking


# Renderers whose template doesn't drop past-asst thinking or has no
# place to re-emit it. For these, override flags MUST be no-ops.
NO_OP_MODELS = {
    "deepseek-ai/DeepSeek-V3",
    "deepseek-ai/DeepSeek-V3-Base",
    "moonshotai/Kimi-K2-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
}


CONVERSATION = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "I should call the weather tool for Paris.",
        "content": "Let me check.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "name": "get_weather", "content": "Sunny, 22C"},
    {
        "role": "assistant",
        "reasoning_content": "The tool returned the weather.",
        "content": "Sunny, 22C in Paris.",
    },
    {"role": "user", "content": "And Berlin?"},
]


def test_should_preserve_past_thinking_classification():
    # CURRENT-block-only behaviour. between_tool_calls preserves thinking
    # ONLY for asst messages that sit AFTER the last user turn AND are in
    # a segment that contains a tool. Anything before the last user turn
    # falls back to template default (typically dropped).

    # Live tool cycle: U-A_tc-T-A_final, no trailing user. The whole
    # post-user segment contains a tool, so both A's are preserved.
    live_cycle = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "reasoning_content": "r1",
            "tool_calls": [{"function": {"name": "f", "arguments": {}}}],
        },
        {"role": "tool", "name": "f", "content": "data"},
        {"role": "assistant", "reasoning_content": "r2", "content": "answer"},
    ]
    assert should_preserve_past_thinking(
        live_cycle,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    assert should_preserve_past_thinking(
        live_cycle,
        3,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )

    # Same shape with a NEW user appended → now the prior tool block is
    # "older" and between_tool_calls must drop its thinking (template
    # default). Only preserve_all_thinking would keep them.
    closed_cycle = live_cycle + [{"role": "user", "content": "next"}]
    assert not should_preserve_past_thinking(
        closed_cycle,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    assert not should_preserve_past_thinking(
        closed_cycle,
        3,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )
    # preserve_all_thinking still keeps them.
    assert should_preserve_past_thinking(
        closed_cycle,
        1,
        preserve_all_thinking=True,
        preserve_thinking_between_tool_calls=False,
    )
    assert should_preserve_past_thinking(
        closed_cycle,
        3,
        preserve_all_thinking=True,
        preserve_thinking_between_tool_calls=False,
    )

    # Current segment without a tool → not a tool cycle → not preserved.
    no_tool_yet = [
        {"role": "user", "content": "q"},
        {"role": "assistant", "reasoning_content": "r", "content": "a"},
    ]
    assert not should_preserve_past_thinking(
        no_tool_yet,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=True,
    )

    # Both flags False → always False.
    assert not should_preserve_past_thinking(
        live_cycle,
        1,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
    )


def test_preserve_flags_default_unchanged(model_name, tokenizer, renderer):
    # Calling render with the flags explicitly off must be byte-identical
    # to the bare call (defaults).
    bare = renderer.render_ids(CONVERSATION)
    explicit_off = renderer.render_ids(
        CONVERSATION,
        preserve_all_thinking=False,
        preserve_thinking_between_tool_calls=False,
    )
    assert bare == explicit_off, (
        f"{model_name}: explicit flags=False must equal bare default render"
    )


def test_preserve_all_thinking_grows_or_no_op(model_name, tokenizer, renderer):
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    preserved = renderer.render_ids(CONVERSATION, preserve_all_thinking=True)

    if model_name in NO_OP_MODELS:
        assert preserved == default, (
            f"{model_name} is a no-op renderer; preserve_all_thinking must "
            f"not change output (got {len(default)} → {len(preserved)})"
        )
    else:
        assert len(preserved) > len(default), (
            f"{model_name}: preserve_all_thinking should add tokens for a "
            f"conversation with past-asst reasoning_content "
            f"(default={len(default)}, preserved={len(preserved)})"
        )


def test_preserve_between_tool_calls_strict_subset(model_name, tokenizer, renderer):
    """``preserve_thinking_between_tool_calls`` is strictly weaker than
    ``preserve_all_thinking``: token count satisfies default <= between <= all."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    default = renderer.render_ids(CONVERSATION)
    between = renderer.render_ids(
        CONVERSATION, preserve_thinking_between_tool_calls=True
    )
    all_ = renderer.render_ids(CONVERSATION, preserve_all_thinking=True)
    assert len(default) <= len(between) <= len(all_), (
        f"{model_name}: expected default <= between <= all, "
        f"got {len(default)} <= {len(between)} <= {len(all_)}"
    )


LIVE_TOOL_CYCLE = [
    {"role": "user", "content": "Weather in Paris?"},
    {
        "role": "assistant",
        "reasoning_content": "Let me call the tool.",
        "content": "Calling.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}
        ],
    },
    {"role": "tool", "name": "get_weather", "content": "Sunny, 22C"},
    {
        "role": "assistant",
        "reasoning_content": "Tool returned weather.",
        "content": "Sunny.",
    },
]


def test_preserve_btc_on_live_cycle_matches_all(model_name, tokenizer, renderer):
    """In a live tool cycle (no trailing user), every past-asst sits in
    the current tool-bearing segment. ``preserve_thinking_between_tool_calls``
    should preserve all of their thinking — same set of asst messages as
    ``preserve_all_thinking``, so the resulting token sequences must be
    identical (independent of which template-default condition each
    renderer uses internally)."""
    from renderers.default import DefaultRenderer

    if isinstance(renderer, DefaultRenderer):
        pytest.skip("DefaultRenderer raises on these flags — covered separately")
    btc = renderer.render_ids(
        LIVE_TOOL_CYCLE, preserve_thinking_between_tool_calls=True
    )
    all_ = renderer.render_ids(LIVE_TOOL_CYCLE, preserve_all_thinking=True)
    assert btc == all_, (
        f"{model_name}: in a live tool cycle btc must match preserve_all "
        f"(got len(btc)={len(btc)}, len(all)={len(all_)})"
    )


def test_default_renderer_raises_on_flags():
    """``DefaultRenderer`` falls back to apply_chat_template with no
    selective re-emit pathway, so it must raise rather than silently ignore."""
    from transformers import AutoTokenizer

    from renderers import create_renderer

    tok = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True
    )
    renderer = create_renderer(tok, renderer="default")
    with pytest.raises(NotImplementedError):
        renderer.render_ids(CONVERSATION, preserve_all_thinking=True)
    with pytest.raises(NotImplementedError):
        renderer.render_ids(CONVERSATION, preserve_thinking_between_tool_calls=True)
