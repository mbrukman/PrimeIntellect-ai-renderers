# Offline Renderer Inference Examples

Each recipe keeps chat templating in `renderers` and sends token IDs to the
backend:

1. Load a Hugging Face tokenizer.
2. Build a model-specific `Renderer`.
3. Render chat messages to prompt token IDs locally.
4. Pass token IDs directly to an offline inference engine.
5. Parse completion token IDs with the same renderer.
6. Bridge the next turn without re-rendering prior assistant output.

The scripts use PEP 723 `uv` headers, so backend dependencies stay local to the
recipe and do not touch the repo `uv.lock`.

## vLLM Multi-Turn Recipe

```bash
CUDA_VISIBLE_DEVICES=0 uv run --script examples/vllm/multiturn_generate_vllm.py
```

The vLLM script targets `vllm>=0.20` and uses `prompt_token_ids`, so vLLM
does not apply a chat template.

## SGLang Multi-Turn Recipe

```bash
CUDA_VISIBLE_DEVICES=1 uv run --script examples/sglang/multiturn_generate_sglang.py
```

The SGLang script uses `input_ids`, so SGLang does not apply a chat template.
It leaves `openai-harmony` at SGLang's pinned version for dependency resolution.

The script sets the SGLang Blackwell workaround needed for Qwen3.5
(`attention_backend="triton"` and `SGLANG_DISABLE_CUDNN_CHECK=1`) inline so the
recipe stays focused on the renderer flow.

## Two-GPU Validation

Run the recipes in parallel, one backend per GPU:

```bash
CUDA_VISIBLE_DEVICES=0 uv run --script examples/vllm/multiturn_generate_vllm.py \
  --max-new-tokens 512 &

CUDA_VISIBLE_DEVICES=1 uv run --script examples/sglang/multiturn_generate_sglang.py \
  --max-new-tokens 512 &

wait
```

Both scripts run `Qwen/Qwen3.5-4B` with `enable_thinking=True` and `False`, then
`openai/gpt-oss-20b`.

## Multimodal Note

Renderers are text-only today. For image/video demos, use the backend's message
or prompt path until renderers grow multimodal placeholder support.
