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

__all__ = [
    "attention_entropy",
    "attention_sparsity",
    "explain_attention",
    "extract_attention_weights",
    "rollout_attention",
    "explain_with_gradients",
    "gradient_x_input",
    "integrated_gradients",
    "occlusion_sensitivity",
    "saliency_map",
    "smooth_gradients",
]
