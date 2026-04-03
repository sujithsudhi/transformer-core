"""
Gradient-based explainability methods for transformer models.

This module provides implementations of integrated gradients, saliency maps,
and other gradient attribution techniques.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor


def _clone_with_gradients(inputs: Tensor) -> Tensor:
    return inputs.detach().clone().requires_grad_(True)


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
        if outputs.dim() < 2:
            raise ValueError("Integer targets require outputs with at least two dimensions.")
        selected = outputs.select(dim=-1, index=target)
        return selected.reshape(batch_size, -1).sum(dim=-1)

    if not isinstance(target, Tensor):
        target = torch.as_tensor(target, device=outputs.device)
    else:
        target = target.to(outputs.device)

    if target.dim() == 0:
        if outputs.dim() < 2:
            raise ValueError("Scalar targets require outputs with at least two dimensions.")
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


def _set_eval_mode(model: torch.nn.Module) -> bool:
    was_training = model.training
    model.eval()
    return was_training


def _restore_training_mode(model: torch.nn.Module, was_training: bool) -> None:
    if was_training:
        model.train()


def integrated_gradients(
    model: torch.nn.Module,
    inputs: Tensor,
    target: Optional[Tensor] = None,
    baseline: Optional[Tensor] = None,
    steps: int = 50,
    return_convergence_delta: bool = False
) -> Tuple[Tensor, Optional[Tensor]]:
    """
    Compute Integrated Gradients attribution.

    Based on "Axiomatic Attribution for Deep Networks" by Sundararajan et al. (2017).

    Args:
        model: The model to explain
        inputs: Input tensor requiring gradients
        target: Target output for attribution (if None, uses argmax for logits)
        baseline: Baseline input (if None, uses zero tensor)
        steps: Number of interpolation steps
        return_convergence_delta: Whether to return the completeness gap per sample

    Returns:
        Attribution tensor and optional convergence delta
    """
    if steps <= 0:
        raise ValueError("steps must be a positive integer.")

    was_training = _set_eval_mode(model)
    inputs_for_attr = inputs.detach()

    if baseline is None:
        baseline = torch.zeros_like(inputs_for_attr)
    baseline = baseline.to(device=inputs_for_attr.device, dtype=inputs_for_attr.dtype)

    with torch.no_grad():
        input_outputs = model(inputs_for_attr)
        resolved_target = _infer_default_target(input_outputs, target)
        baseline_outputs = model(baseline)

    total_gradients = torch.zeros_like(inputs_for_attr)

    for step in range(1, steps + 1):
        alpha = step / steps
        interpolated = (baseline + alpha * (inputs_for_attr - baseline)).detach().requires_grad_(True)
        outputs = model(interpolated)
        objective = _compute_batch_objective(outputs, resolved_target)
        model.zero_grad()
        gradients = torch.autograd.grad(objective.sum(), interpolated)[0]
        total_gradients += gradients

    attributions = (inputs_for_attr - baseline) * total_gradients / steps
    convergence_delta = None

    if return_convergence_delta:
        input_scores = _compute_batch_objective(input_outputs, resolved_target)
        baseline_scores = _compute_batch_objective(baseline_outputs, resolved_target)
        completeness = attributions.reshape(attributions.shape[0], -1).sum(dim=-1)
        convergence_delta = (input_scores - baseline_scores - completeness).abs().detach()

    _restore_training_mode(model, was_training)
    return attributions.detach(), convergence_delta


def saliency_map(
    model: torch.nn.Module,
    inputs: Tensor,
    target: Optional[Tensor] = None
) -> Tensor:
    """
    Compute saliency map (gradient magnitude).

    Args:
        model: The model to explain
        inputs: Input tensor
        target: Target output (if None, uses argmax for logits)

    Returns:
        Saliency map tensor
    """
    was_training = _set_eval_mode(model)
    inputs_for_grad = _clone_with_gradients(inputs)

    outputs = model(inputs_for_grad)
    resolved_target = _infer_default_target(outputs, target)
    objective = _compute_batch_objective(outputs, resolved_target)
    model.zero_grad()
    gradients = torch.autograd.grad(objective.sum(), inputs_for_grad)[0]

    _restore_training_mode(model, was_training)
    return gradients.abs().detach()


def gradient_x_input(
    model: torch.nn.Module,
    inputs: Tensor,
    target: Optional[Tensor] = None
) -> Tensor:
    """
    Compute Gradients x Input attribution.

    Args:
        model: The model to explain
        inputs: Input tensor
        target: Target output

    Returns:
        Attribution tensor
    """
    was_training = _set_eval_mode(model)
    inputs_for_grad = _clone_with_gradients(inputs)

    outputs = model(inputs_for_grad)
    resolved_target = _infer_default_target(outputs, target)
    objective = _compute_batch_objective(outputs, resolved_target)
    model.zero_grad()
    gradients = torch.autograd.grad(objective.sum(), inputs_for_grad)[0]

    _restore_training_mode(model, was_training)
    return (gradients * inputs_for_grad).detach()


def smooth_gradients(
    model: torch.nn.Module,
    inputs: Tensor,
    target: Optional[Tensor] = None,
    noise_level: float = 0.1,
    num_samples: int = 50
) -> Tensor:
    """
    Compute SmoothGrad attribution.

    Based on "SmoothGrad: removing noise by adding noise" by Smilkov et al. (2017).

    Args:
        model: The model to explain
        inputs: Input tensor
        target: Target output
        noise_level: Standard deviation of noise
        num_samples: Number of noisy samples

    Returns:
        SmoothGrad attribution
    """
    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")

    with torch.no_grad():
        base_outputs = model(inputs.detach())
        resolved_target = _infer_default_target(base_outputs, target)

    attributions = []
    for _ in range(num_samples):
        noisy_inputs = inputs + torch.randn_like(inputs) * noise_level
        attributions.append(saliency_map(model, noisy_inputs, resolved_target))

    return torch.stack(attributions).mean(dim=0)


def occlusion_sensitivity(
    model: torch.nn.Module,
    inputs: Tensor,
    target: Optional[Tensor] = None,
    occlusion_size: int = 1,
    stride: int = 1,
    baseline_value: float = 0.0
) -> Tensor:
    """
    Compute occlusion sensitivity over sequence positions.

    Args:
        model: The model to explain
        inputs: Input tensor of shape (batch, seq_len, embed_dim)
        target: Target output
        occlusion_size: Number of consecutive tokens to occlude
        stride: Stride for occlusion window
        baseline_value: Value to use for occluded tokens

    Returns:
        Attribution scores for each position
    """
    if inputs.dim() != 3:
        raise ValueError("occlusion_sensitivity expects inputs of shape (batch, seq_len, embed_dim).")
    if occlusion_size <= 0 or stride <= 0:
        raise ValueError("occlusion_size and stride must be positive integers.")

    was_training = _set_eval_mode(model)
    batch_size, seq_len, _ = inputs.shape

    with torch.no_grad():
        baseline_output = model(inputs)
        resolved_target = _infer_default_target(baseline_output, target)
        baseline_score = _compute_batch_objective(baseline_output, resolved_target)

    attributions = torch.zeros(batch_size, seq_len, device=inputs.device)
    coverage = torch.zeros(seq_len, device=inputs.device)

    for start_pos in range(0, seq_len - occlusion_size + 1, stride):
        end_pos = start_pos + occlusion_size
        occluded_inputs = inputs.clone()
        occluded_inputs[:, start_pos:end_pos, :] = baseline_value

        with torch.no_grad():
            occluded_output = model(occluded_inputs)
            occluded_score = _compute_batch_objective(occluded_output, resolved_target)

        delta = baseline_score - occluded_score
        attributions[:, start_pos:end_pos] += delta.unsqueeze(-1).expand(-1, occlusion_size)
        coverage[start_pos:end_pos] += 1

    attributions = attributions / coverage.clamp_min(1).unsqueeze(0)
    _restore_training_mode(model, was_training)
    return attributions.detach()


def explain_with_gradients(
    model: torch.nn.Module,
    inputs: Tensor,
    method: str = "integrated_gradients",
    target: Optional[Tensor] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Unified interface for gradient-based explanation methods.

    Args:
        model: The model to explain
        inputs: Input tensor
        method: Explanation method ("integrated_gradients", "saliency", "grad_x_input", "smooth_grad", "occlusion")
        target: Target output
        **kwargs: Additional arguments for the method

    Returns:
        Dictionary with attributions and metadata
    """
    method_funcs = {
        "integrated_gradients": integrated_gradients,
        "saliency": saliency_map,
        "grad_x_input": gradient_x_input,
        "smooth_grad": smooth_gradients,
        "occlusion": occlusion_sensitivity,
    }

    if method not in method_funcs:
        raise ValueError(f"Unknown method: {method}. Available: {list(method_funcs.keys())}")

    func = method_funcs[method]
    if method == "integrated_gradients":
        attribution, convergence_delta = func(model, inputs, target, **kwargs)
        result = {
            "attribution": attribution,
            "method": method,
            "convergence_delta": convergence_delta,
        }
    else:
        attribution = func(model, inputs, target, **kwargs)
        result = {
            "attribution": attribution,
            "method": method,
        }

    if inputs.dim() == 3:
        result["token_importance"] = attribution.abs().sum(dim=-1)
    elif attribution.dim() == 2:
        result["token_importance"] = attribution.abs()

    return result
