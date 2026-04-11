"""
Perturbation-based faithfulness metrics for transformer explanations.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from transformer_core.xai.multimodal import MultimodalLayout, select_modality_scores
from transformer_core.xai.results import FaithfulnessResult, VLMFaithfulnessResult


def _infer_default_target(outputs: Tensor, target: Optional[Tensor | int]) -> Optional[Tensor | int]:
    if target is not None:
        return target
    if outputs.dim() == 2:
        return outputs.argmax(dim=-1)
    return None


def _compute_batch_objective(outputs: Tensor, target: Optional[Tensor | int]) -> Tensor:
    if outputs.dim() == 0:
        return outputs.reshape(1)

    batch_size = outputs.shape[0]

    if target is None:
        reduce_dims = tuple(range(1, outputs.dim()))
        return outputs.sum(dim=reduce_dims) if reduce_dims else outputs

    if isinstance(target, int):
        selected = outputs.select(dim=-1, index=target)
        return selected.reshape(batch_size, -1).sum(dim=-1)

    if not isinstance(target, Tensor):
        target = torch.as_tensor(target, device=outputs.device)
    else:
        target = target.to(outputs.device)

    if target.dim() == 0:
        selected = outputs.select(dim=-1, index=int(target.item()))
        return selected.reshape(batch_size, -1).sum(dim=-1)

    if outputs.dim() == 2 and target.shape == (batch_size,):
        indices = target.to(dtype=torch.long)
        return outputs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)

    if outputs.dim() >= 3 and target.shape == outputs.shape[:-1]:
        indices = target.to(dtype=torch.long)
        gathered = outputs.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
        return gathered.reshape(batch_size, -1).sum(dim=-1)

    if target.shape == outputs.shape:
        return (outputs * target).reshape(batch_size, -1).sum(dim=-1)

    raise ValueError(
        "Target shape is incompatible with model outputs. "
        f"Got target shape {tuple(target.shape)} and outputs shape {tuple(outputs.shape)}."
    )


def select_topk_positions(token_scores: Tensor, top_k: int) -> list[int]:
    """Return the top-k positions from a single-example score vector."""

    if token_scores.dim() != 1:
        raise ValueError("select_topk_positions expects a 1D score tensor.")
    if top_k <= 0:
        return []
    top_k = min(top_k, token_scores.numel())
    return torch.topk(token_scores, k=top_k).indices.tolist()


def ablate_positions(inputs: Tensor, positions: list[int], baseline_value: float = 0.0) -> Tensor:
    """Replace the selected sequence positions with a baseline value."""

    ablated = inputs.clone()
    if positions:
        ablated[:, positions, :] = baseline_value
    return ablated


def keep_only_positions(inputs: Tensor, positions: list[int], baseline_value: float = 0.0) -> Tensor:
    """Keep the selected sequence positions and baseline everything else."""

    kept = torch.full_like(inputs, baseline_value)
    if positions:
        kept[:, positions, :] = inputs[:, positions, :]
    return kept


def comprehensiveness(
    model: torch.nn.Module,
    inputs: Tensor,
    positions: list[int],
    *,
    target: Optional[Tensor] = None,
    baseline_value: float = 0.0,
) -> Tensor:
    """How much the prediction drops when important positions are removed."""

    with torch.no_grad():
        baseline_outputs = model(inputs)
        resolved_target = _infer_default_target(baseline_outputs, target)
        baseline_score = _compute_batch_objective(baseline_outputs, resolved_target)
        ablated_score = _compute_batch_objective(
            model(ablate_positions(inputs, positions, baseline_value=baseline_value)),
            resolved_target,
        )
    return baseline_score - ablated_score


def sufficiency(
    model: torch.nn.Module,
    inputs: Tensor,
    positions: list[int],
    *,
    target: Optional[Tensor] = None,
    baseline_value: float = 0.0,
) -> Tensor:
    """How much of the prediction is preserved by keeping only the selected positions."""

    with torch.no_grad():
        baseline_outputs = model(inputs)
        resolved_target = _infer_default_target(baseline_outputs, target)
        baseline_score = _compute_batch_objective(baseline_outputs, resolved_target)
        kept_score = _compute_batch_objective(
            model(keep_only_positions(inputs, positions, baseline_value=baseline_value)),
            resolved_target,
        )
    return baseline_score - kept_score


def evaluate_vlm_faithfulness(
    model: torch.nn.Module,
    inputs: Tensor,
    *,
    layout: MultimodalLayout,
    token_scores: Tensor,
    top_k_text: int = 3,
    top_k_image: int = 3,
    target: Optional[Tensor] = None,
    baseline_value: float = 0.0,
) -> VLMFaithfulnessResult:
    """
    Evaluate text-only, image-only, and joint faithfulness for a single example.
    """
    if token_scores.dim() != 1:
        raise ValueError("evaluate_vlm_faithfulness expects a 1D score tensor for a single example.")

    text_positions = layout.positions("text_token")
    image_positions = layout.positions("image_patch")
    text_scores = select_modality_scores(token_scores, layout, "text_token")
    image_scores = select_modality_scores(token_scores, layout, "image_patch")

    selected_text_rel = select_topk_positions(text_scores.abs(), top_k_text)
    selected_image_rel = select_topk_positions(image_scores.abs(), top_k_image)
    selected_text = [text_positions[index] for index in selected_text_rel]
    selected_image = [image_positions[index] for index in selected_image_rel]
    selected_overall = sorted(set(selected_text + selected_image))

    overall = FaithfulnessResult(
        selected_positions=selected_overall,
        comprehensiveness=comprehensiveness(
            model,
            inputs,
            selected_overall,
            target=target,
            baseline_value=baseline_value,
        ),
        sufficiency=sufficiency(
            model,
            inputs,
            selected_overall,
            target=target,
            baseline_value=baseline_value,
        ),
        metadata={"scope": "overall"},
    )

    text = FaithfulnessResult(
        selected_positions=selected_text,
        comprehensiveness=comprehensiveness(
            model,
            inputs,
            selected_text,
            target=target,
            baseline_value=baseline_value,
        ),
        sufficiency=sufficiency(
            model,
            inputs,
            selected_text,
            target=target,
            baseline_value=baseline_value,
        ),
        metadata={"scope": "text"},
    ) if selected_text else None

    image = FaithfulnessResult(
        selected_positions=selected_image,
        comprehensiveness=comprehensiveness(
            model,
            inputs,
            selected_image,
            target=target,
            baseline_value=baseline_value,
        ),
        sufficiency=sufficiency(
            model,
            inputs,
            selected_image,
            target=target,
            baseline_value=baseline_value,
        ),
        metadata={"scope": "image"},
    ) if selected_image else None

    return VLMFaithfulnessResult(overall=overall, text=text, image=image)
