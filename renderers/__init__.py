try:
    from renderers._version import __version__
except ImportError:
    # Source checkout without a built artifact (e.g. editable install
    # before the first ``uv build`` populates ``_version.py``). Real
    # installs always have it.
    __version__ = "0+unknown"

from renderers.base import (
    Content,
    ContentPart,
    ImagePart,
    MULTIMODAL_MODELS,
    Message,
    MultiModalData,
    MultimodalRenderer,
    ParsedResponse,
    ParsedToolCall,
    PlaceholderRange,
    Processor,
    RenderedConversation,
    RenderedTokens,
    Renderer,
    RendererPool,
    TextPart,
    ThinkingPart,
    Tokenizer,
    ToolCall,
    ToolCallFunction,
    ToolCallParseStatus,
    ToolSpec,
    VideoPart,
    attribute_text_segments,
    build_training_sample,
    build_trajectory_step,
    create_renderer,
    create_renderer_pool,
    is_multimodal,
    reject_assistant_in_extension,
    trim_to_turn_close,
)
from renderers.configs import (
    AutoRendererConfig,
    BaseRendererConfig,
    config_from_name,
    DefaultRendererConfig,
    DeepSeekV3RendererConfig,
    GLM45RendererConfig,
    GLM51RendererConfig,
    GLM5RendererConfig,
    GptOssRendererConfig,
    KimiK25RendererConfig,
    KimiK2RendererConfig,
    LagunaXS2RendererConfig,
    MiniMaxM2RendererConfig,
    Nemotron3RendererConfig,
    Qwen35RendererConfig,
    Qwen36RendererConfig,
    Qwen3RendererConfig,
    Qwen3VLRendererConfig,
    RendererConfig,
)

# Concrete renderer classes are lazy-loaded so that consumers needing
# only the config layer (``RendererConfig`` discriminated union) don't
# pay the cost of importing every renderer module up front. ``__getattr__``
# (PEP 562) resolves the names on first attribute access, so ``from
# renderers import DefaultRenderer`` and ``renderers.DefaultRenderer`` both
# work transparently. ``create_renderer`` doesn't depend on these eager
# imports — ``renderers.base._populate_registry`` lazy-imports the concrete
# classes itself when a renderer is instantiated.
#
# As of issue #31, ``transformers`` is an optional extra: the renderer
# modules type their ``tokenizer`` / ``processor`` params against the
# ``Tokenizer`` / ``Processor`` protocols in ``renderers.base`` rather than
# ``transformers.PreTrainedTokenizer``, so ``import renderers`` (and
# constructing a text renderer with your own tokenizer) no longer pulls in
# ``transformers`` at all. It's loaded lazily only by ``load_tokenizer`` /
# ``create_renderer*`` and the VLM renderers — see ``_require_transformers``.
#
# ``renderers.client`` (the vLLM ``/inference/v1/generate`` client) is
# likewise opt-in: it depends on the ``openai`` SDK + ``httpx`` (the
# ``renderers[vllm]`` extra) and is deliberately *not* imported here, so
# ``import renderers`` stays free of HTTP/engine deps. Import it explicitly
# (``from renderers.client import generate, OverlongPromptError``) when you
# want it.
_LAZY_RENDERERS: dict[str, str] = {
    "DeepSeekV3Renderer": "renderers.deepseek_v3",
    "DefaultRenderer": "renderers.default",
    "GLM45Renderer": "renderers.glm45",
    "GLM51Renderer": "renderers.glm5",
    "GLM5Renderer": "renderers.glm5",
    "GptOssRenderer": "renderers.gpt_oss",
    "KimiK25Renderer": "renderers.kimi_k25",
    "KimiK2Renderer": "renderers.kimi_k2",
    "LagunaXS2Renderer": "renderers.laguna_xs2",
    "MiniMaxM2Renderer": "renderers.minimax_m2",
    "Nemotron3Renderer": "renderers.nemotron3",
    "Qwen35Renderer": "renderers.qwen35",
    "Qwen36Renderer": "renderers.qwen36",
    "Qwen3Renderer": "renderers.qwen3",
    "Qwen3VLRenderer": "renderers.qwen3_vl",
}


def __getattr__(name: str):
    if name in _LAZY_RENDERERS:
        import importlib

        module = importlib.import_module(_LAZY_RENDERERS[name])
        value = getattr(module, name)
        globals()[name] = value  # cache for subsequent lookups
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals().keys()) | set(_LAZY_RENDERERS))


__all__ = [
    "AutoRendererConfig",
    "BaseRendererConfig",
    "Content",
    "ContentPart",
    "DeepSeekV3Renderer",
    "DeepSeekV3RendererConfig",
    "DefaultRenderer",
    "DefaultRendererConfig",
    "GLM45Renderer",
    "GLM45RendererConfig",
    "GLM51Renderer",
    "GLM51RendererConfig",
    "GLM5Renderer",
    "GLM5RendererConfig",
    "GptOssRenderer",
    "GptOssRendererConfig",
    "ImagePart",
    "KimiK25Renderer",
    "KimiK25RendererConfig",
    "KimiK2Renderer",
    "KimiK2RendererConfig",
    "LagunaXS2Renderer",
    "LagunaXS2RendererConfig",
    "MULTIMODAL_MODELS",
    "Message",
    "MiniMaxM2Renderer",
    "MiniMaxM2RendererConfig",
    "MultiModalData",
    "MultimodalRenderer",
    "Nemotron3Renderer",
    "Nemotron3RendererConfig",
    "ParsedResponse",
    "ParsedToolCall",
    "PlaceholderRange",
    "Processor",
    "Qwen35Renderer",
    "Qwen35RendererConfig",
    "Qwen36Renderer",
    "Qwen36RendererConfig",
    "Qwen3Renderer",
    "Qwen3RendererConfig",
    "Qwen3VLRenderer",
    "Qwen3VLRendererConfig",
    "RenderedConversation",
    "RenderedTokens",
    "Renderer",
    "RendererConfig",
    "RendererPool",
    "TextPart",
    "ThinkingPart",
    "Tokenizer",
    "ToolCall",
    "ToolCallFunction",
    "ToolCallParseStatus",
    "ToolSpec",
    "VideoPart",
    "__version__",
    "attribute_text_segments",
    "build_training_sample",
    "build_trajectory_step",
    "config_from_name",
    "create_renderer",
    "create_renderer_pool",
    "is_multimodal",
    "reject_assistant_in_extension",
    "trim_to_turn_close",
]
