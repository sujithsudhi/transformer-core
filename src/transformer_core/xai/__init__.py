# transformer_core.xai
# Explainable AI tools for transformer models

from .attention import (
    attention_entropy,
    attention_sparsity,
    explain_attention,
    extract_attention_weights,
    rollout_attention,
)
from .gradients import (
    explain_with_gradients,
    gradient_x_input,
    integrated_gradients,
    occlusion_sensitivity,
    saliency_map,
    smooth_gradients,
)
from .multimodal import (
    ModalitySpan,
    MultimodalLayout,
    PatchGrid,
    build_vlm_layout,
    infer_patch_grid,
)
from .results import AttributionResult, AttentionTraceResult, VLMExplanationResult

__all__ = [
    "AttentionTraceResult",
    "AttributionResult",
    "ModalitySpan",
    "MultimodalLayout",
    "PatchGrid",
    "VLMExplanationResult",
    "attention_entropy",
    "attention_sparsity",
    "build_vlm_layout",
    "explain_attention",
    "extract_attention_weights",
    "infer_patch_grid",
    "rollout_attention",
    "explain_with_gradients",
    "gradient_x_input",
    "integrated_gradients",
    "occlusion_sensitivity",
    "saliency_map",
    "smooth_gradients",
]
