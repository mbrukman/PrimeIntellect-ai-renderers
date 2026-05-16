"""Per-token ``RenderedTokens.content_mask`` invariants."""

from __future__ import annotations


def test_content_mask_length_or_empty(model_name, renderer):
    msgs = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    rendered = renderer.render(msgs)
    n_tokens = len(rendered.token_ids)
    n_mask = len(rendered.content_mask)
    assert n_mask == 0 or n_mask == n_tokens, (
        f"{model_name}: content_mask length {n_mask} must be 0 or match "
        f"token_ids length {n_tokens}"
    )


def test_content_mask_excludes_role_and_generation_prompt_tokens(model_name, renderer):
    msgs = [{"role": "user", "content": "Hi"}]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    if not rendered.content_mask:
        return

    bad_generation = [
        k
        for k, (msg_idx, is_content) in enumerate(
            zip(rendered.message_indices, rendered.content_mask)
        )
        if msg_idx == -1 and is_content
    ]
    assert not bad_generation, (
        f"{model_name}: generation prompt tokens marked as content at "
        f"positions {bad_generation[:8]}"
    )

    user_positions = [k for k, idx in enumerate(rendered.message_indices) if idx == 0]
    assert user_positions, f"{model_name}: no tokens attributed to user message"
    assert not rendered.content_mask[user_positions[0]], (
        f"{model_name}: first user-attributed token at k={user_positions[0]} "
        "should be role/control scaffolding, not content"
    )
    assert any(rendered.content_mask[k] for k in user_positions), (
        f"{model_name}: no user message tokens marked as content"
    )


def test_content_mask_marks_tool_text_but_not_tool_wrappers(model_name, renderer):
    msgs = [
        {"role": "user", "content": "Use the tool."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": {"query": "x"}},
                }
            ],
        },
        {"role": "tool", "content": "tool result"},
    ]
    rendered = renderer.render(msgs, add_generation_prompt=True)
    if not rendered.content_mask:
        return

    tool_positions = [k for k, idx in enumerate(rendered.message_indices) if idx == 2]
    if not tool_positions:
        return

    assert not rendered.content_mask[tool_positions[0]], (
        f"{model_name}: first tool-attributed token at k={tool_positions[0]} "
        "should be wrapper scaffolding, not content"
    )
    assert any(rendered.content_mask[k] for k in tool_positions), (
        f"{model_name}: tool message has no content tokens marked"
    )
