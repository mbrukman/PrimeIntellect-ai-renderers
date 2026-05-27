"""``transformers`` is an optional extra (issue #31).

These tests prove the boundary holds: importing ``renderers`` and driving a
text renderer with a bring-your-own tokenizer must work with ``transformers``
(and ``fastokens``) absent, and the convenience helpers that *do* need it must
fail with a clear, actionable error.

The dev environment has ``transformers`` installed, so we simulate its absence
in a subprocess by installing a meta-path finder that raises ``ImportError``
for ``transformers`` / ``fastokens`` (and their submodules) before anything
imports ``renderers``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

# Shared preamble: block ``transformers`` / ``fastokens`` at import time, the
# way a lightweight install (no extra) would behave.
_BLOCK_PREAMBLE = """
import sys
import importlib.abc
import importlib.machinery

_BLOCKED = ("transformers", "fastokens")


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        root = name.split(".")[0]
        if root in _BLOCKED:
            raise ImportError(f"{name} is blocked (simulating optional extra)")
        return None


sys.meta_path.insert(0, _Blocker())

# A minimal tokenizer satisfying renderers.Tokenizer with no HF dependency.
# Special tokens map to fixed ids; ordinary text is char-level (ord + an
# offset that can't collide with the special ids). decode inverts encode, and
# __call__ supports return_offsets_mapping so attribute_text_segments uses this
# tokenizer directly rather than falling back to a vanilla HF one.
_CHAR_BASE = 200_000


class FakeTokenizer:
    name_or_path = "fake/qwen3-text"
    unk_token_id = 0

    def __init__(self):
        specials = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ]
        self._special = {t: 150_000 + i for i, t in enumerate(specials)}
        self._rev = {v: k for k, v in self._special.items()}
        self.eos_token_id = self._special["<|im_end|>"]

    def convert_tokens_to_ids(self, token):
        return self._special.get(token, self.unk_token_id)

    def encode(self, text, add_special_tokens=False, **kw):
        return [_CHAR_BASE + ord(c) for c in text]

    def decode(self, ids, **kw):
        out = []
        for i in ids:
            if i in self._rev:
                out.append(self._rev[i])
            elif i >= _CHAR_BASE:
                out.append(chr(i - _CHAR_BASE))
        return "".join(out)

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False, **kw):
        ids = [_CHAR_BASE + ord(c) for c in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result
"""


def _run(body: str) -> subprocess.CompletedProcess:
    script = _BLOCK_PREAMBLE + textwrap.dedent(body)
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )


def test_text_renderer_works_without_transformers():
    """Import renderers + drive a text renderer with no transformers present."""
    proc = _run(
        """
        import renderers
        from renderers import Qwen3Renderer

        tok = FakeTokenizer()
        r = Qwen3Renderer(tok)

        prompt_ids = r.render_ids(
            [{"role": "user", "content": "hi there"}],
            add_generation_prompt=True,
        )
        assert prompt_ids and all(isinstance(i, int) for i in prompt_ids)

        # parse a hand-built assistant completion: text then the stop token.
        completion = tok.encode("Hello!") + [tok.convert_tokens_to_ids("<|im_end|>")]
        parsed = r.parse_response(completion)
        assert parsed.content == "Hello!", parsed.content

        # The whole point: transformers / fastokens never got imported.
        leaked = [m for m in sys.modules if m.split(".")[0] in _BLOCKED]
        assert not leaked, f"unexpected import: {leaked}"
        print("OK")
        """
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_import_renderers_without_client_deps():
    """``renderers.client`` (vLLM generate client) is opt-in via the ``[vllm]``
    extra — ``import renderers`` and driving a renderer must not pull in the
    ``openai`` SDK or ``httpx``."""
    proc = _run(
        """
        # Additionally block the client's deps.
        _BLOCKED = _BLOCKED + ("openai", "httpx")

        import renderers
        from renderers import Qwen3Renderer

        r = Qwen3Renderer(FakeTokenizer())
        ids = r.render_ids([{"role": "user", "content": "hi"}], add_generation_prompt=True)
        assert ids

        leaked = [m for m in sys.modules if m.split(".")[0] in ("openai", "httpx")]
        assert not leaked, f"client deps leaked into import: {leaked}"
        print("OK")
        """
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in proc.stdout


def test_load_tokenizer_errors_clearly_without_transformers():
    """The convenience helper must point at the extra, not raise a bare
    ``No module named 'transformers'``."""
    proc = _run(
        """
        from renderers.base import load_tokenizer
        try:
            load_tokenizer("Qwen/Qwen3-8B")
        except ImportError as exc:
            assert "renderers[transformers]" in str(exc), str(exc)
            print("OK")
        else:
            raise AssertionError("expected ImportError")
        """
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "OK" in proc.stdout
