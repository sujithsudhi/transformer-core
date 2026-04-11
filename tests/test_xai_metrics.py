"""Tests for faithfulness metrics."""

import torch

from transformer_core.vision.patch_embedding import PatchEmbedding
from transformer_core.xai.metrics import (
    ablate_positions,
    comprehensiveness,
    evaluate_vlm_faithfulness,
    keep_only_positions,
    select_topk_positions,
    sufficiency,
)
from transformer_core.xai.multimodal import build_vlm_layout, infer_patch_grid


class WeightedSequenceClassifier(torch.nn.Module):
    """Deterministic classifier whose score depends on token feature sums."""

    def forward(self, x, mask=None):
        del mask
        score = x[:, :, 0].sum(dim=1)
        return torch.stack([score, -score], dim=-1)


def test_ablate_and_keep_only_positions():
    inputs = torch.arange(24, dtype=torch.float32).view(1, 6, 4)
    ablated = ablate_positions(inputs, [1, 3], baseline_value=-1.0)
    kept = keep_only_positions(inputs, [1, 3], baseline_value=0.0)

    assert torch.all(ablated[:, [1, 3], :] == -1.0)
    assert torch.all(kept[:, [0, 2, 4, 5], :] == 0.0)
    assert torch.equal(kept[:, [1, 3], :], inputs[:, [1, 3], :])


def test_select_topk_positions():
    scores = torch.tensor([0.2, 0.9, 0.1, 0.8])
    assert select_topk_positions(scores, top_k=2) == [1, 3]


def test_comprehensiveness_and_sufficiency():
    model = WeightedSequenceClassifier()
    inputs = torch.zeros(1, 5, 4)
    inputs[:, 1, 0] = 3.0
    inputs[:, 3, 0] = 2.0

    comp = comprehensiveness(model, inputs, [1, 3])
    suff = sufficiency(model, inputs, [1, 3])

    assert torch.allclose(comp, torch.tensor([5.0]))
    assert torch.allclose(suff, torch.tensor([0.0]))


def test_evaluate_vlm_faithfulness():
    patch_grid = infer_patch_grid(PatchEmbedding(image_size=16, patch_size=8, embed_dim=8))
    layout = build_vlm_layout(
        patch_grid=patch_grid,
        num_text_tokens=3,
        cls_tokens=1,
        separator_tokens=1,
    )
    model = WeightedSequenceClassifier()
    inputs = torch.zeros(1, layout.total_length, 4)

    image_positions = layout.positions("image_patch")
    text_positions = layout.positions("text_token")
    inputs[:, image_positions[0], 0] = 4.0
    inputs[:, image_positions[1], 0] = 2.0
    inputs[:, text_positions[0], 0] = 1.5
    inputs[:, text_positions[1], 0] = 1.0

    token_scores = torch.zeros(layout.total_length)
    token_scores[image_positions[0]] = 0.9
    token_scores[image_positions[1]] = 0.8
    token_scores[text_positions[0]] = 0.7
    token_scores[text_positions[1]] = 0.6

    result = evaluate_vlm_faithfulness(
        model,
        inputs,
        layout=layout,
        token_scores=token_scores,
        top_k_text=2,
        top_k_image=2,
    )

    assert result.image is not None
    assert result.text is not None
    assert result.overall.selected_positions == sorted(
        set(result.image.selected_positions + result.text.selected_positions)
    )
    assert torch.allclose(result.image.comprehensiveness, torch.tensor([6.0]))
    assert torch.allclose(result.text.comprehensiveness, torch.tensor([2.5]))
