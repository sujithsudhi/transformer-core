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
from .metrics import (
    ablate_positions,
    comprehensiveness,
    evaluate_vlm_faithfulness,
    keep_only_positions,
    select_topk_positions,
    sufficiency,
)
from .multimodal import (
    ModalitySpan,
    MultimodalLayout,
    PatchGrid,
    build_vlm_layout,
    explain_vlm_attention,
    explain_vlm_with_gradients,
    infer_patch_grid,
    reshape_patch_scores,
    select_modality_scores,
)
from .results import (
    AttributionResult,
    AttentionTraceResult,
    FaithfulnessResult,
    VLMExplanationResult,
    VLMFaithfulnessResult,
)

__all__ = [
    "AttentionTraceResult",
    "AttributionResult",
    "FaithfulnessResult",
    "ModalitySpan",
    "MultimodalLayout",
    "PatchGrid",
    "VLMExplanationResult",
    "VLMFaithfulnessResult",
    "ablate_positions",
    "attention_entropy",
    "attention_sparsity",
    "build_vlm_layout",
    "comprehensiveness",
    "explain_vlm_attention",
    "explain_vlm_with_gradients",
    "explain_attention",
    "extract_attention_weights",
    "evaluate_vlm_faithfulness",
    "infer_patch_grid",
    "keep_only_positions",
    "rollout_attention",
    "reshape_patch_scores",
    "select_topk_positions",
    "select_modality_scores",
    "explain_with_gradients",
    "gradient_x_input",
    "integrated_gradients",
    "occlusion_sensitivity",
    "saliency_map",
    "smooth_gradients",
    "sufficiency",
]
