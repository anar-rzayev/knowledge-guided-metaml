import functools
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F


def mse(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Standard MSE loss on a batch."""
    assert (
        preds.shape == targets.shape
    ), f"Prediction shape {preds.shape} does not match target shape {targets.shape}"
    mse = ((preds - targets) ** 2).mean(0)
    return mse


def full_ranking_bce(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Pairwise binary cross entropy loss between all items in batch.
    
    For each pair of items in the batch, predicts which member has the higher target value.
    Derives logits from prediction pairs as: Pij = preds_i - preds_j
    
    These latent preferences are treated as logits for a binary classifier:
    y_ij = I(targets_i > targets_j)

    Args:
        preds: Tensor of shape (b,) containing predicted values
        targets: Tensor of shape (b,) containing true values
    """
    pairwise_logits = preds[:, None] - preds[None, :]  # b x b
    targets = targets[:, None] > targets[None, :]
    ranking_xent = 0.5 * F.binary_cross_entropy_with_logits(
        pairwise_logits, targets.float(), reduction="none"
    )
    diag_mask = 1 - torch.eye(pairwise_logits.shape[0], device=pairwise_logits.device)
    ranking_xent = (ranking_xent * diag_mask).mean((-1, -2))
    return ranking_xent


def label_smoothed_ranking_loss(
    ranking_loss: Callable,
    preds: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Apply label smoothing to ranking loss.
    
    Label smoothing parameter represents probability of label flip.
    Loss is interpolated between original and flipped predictions.
    
    Args:
        ranking_loss: Base ranking loss function
        preds: Predicted values
        targets: True values
        label_smoothing: Smoothing parameter between 0 and 1
    """
    loss = ranking_loss(preds, targets)
    flipped_loss = ranking_loss(-preds, targets)
    return (1 - label_smoothing) * loss + label_smoothing * flipped_loss


def adaptive_label_smoothed_full_ranking_loss(
    preds: torch.Tensor,
    targets: torch.Tensor,
    label_smoothing_beta: float = 1.0,
) -> torch.Tensor:
    """Adaptive label smoothing based on difference magnitude.
    
    Label smoothing parameter is a function of the magnitude of
    difference in labels.

    Args:
        preds: Predicted values 
        targets: True values
        label_smoothing_beta: Scaling factor for difference-based smoothing
    """
    pairwise_logits = preds[:, None] - preds[None, :]  # b x b
    target_diffs = targets[:, None] - targets[None, :]
    targets = target_diffs > 0
    ranking_xent = 0.5 * F.binary_cross_entropy_with_logits(
        pairwise_logits, targets.float(), reduction="none"
    )
    diag_mask = 1 - torch.eye(pairwise_logits.shape[0], device=pairwise_logits.device)
    ranking_xent = ranking_xent * diag_mask

    flipped_ranking_xent = 0.5 * F.binary_cross_entropy_with_logits(
        pairwise_logits, (~targets).float(), reduction="none"
    )
    flipped_ranking_xent = flipped_ranking_xent * diag_mask

    label_smoothing = torch.sigmoid(-target_diffs.abs() * label_smoothing_beta)

    ranking_xent = (
        1 - label_smoothing
    ) * ranking_xent + label_smoothing * flipped_ranking_xent
    return ranking_xent.mean((-1, -2))


def derangement_vectorized(n: int) -> np.ndarray:
    """Sample random permutation with no fixed points.
    
    Args:
        n: Size of permutation
    Returns:
        Array containing derangement permutation
    """
    assert n > 1
    original = np.arange(n)
    while True:
        shuffled = np.random.permutation(n)
        if np.all(original != shuffled):
            return shuffled


def perm_ranking_bce(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Ranking loss using permuted pairs.
    
    Creates classification tasks by pairing each item with one other.
    Uses difference in predictions as logits to predict which item
    has higher target value.
    
    Args:
        preds: Predicted values
        targets: True values
    """
    perm = derangement_vectorized(preds.shape[0])
    pairwise_logits = preds - preds[perm]
    targets = targets > targets[perm]
    ranking_xent = F.binary_cross_entropy_with_logits(
        pairwise_logits, targets.float(), reduction="mean"
    )
    return ranking_xent


def get_loss(
    loss_name: str, label_smoothing: float = 0.0, label_smoothing_beta: float = 1.0
) -> Callable:
    """Get loss function by name with optional label smoothing.
    
    Args:
        loss_name: Name of loss function
        label_smoothing: Smoothing parameter for ranking losses
        label_smoothing_beta: Beta parameter for adaptive smoothing
        
    Returns:
        Callable loss function
    """
    if loss_name in ["mse", "mse_head"]:
        assert label_smoothing == 0.0, "Label smoothing not compatible with MSE"
        return mse
    elif loss_name == "ranking":
        loss_fn = perm_ranking_bce
        if label_smoothing > 0:
            loss_fn = functools.partial(
                label_smoothed_ranking_loss, loss_fn, label_smoothing=label_smoothing
            )
        return loss_fn
    elif loss_name == "ranking_full":
        loss_fn = full_ranking_bce
        if label_smoothing > 0:
            loss_fn = functools.partial(
                label_smoothed_ranking_loss, loss_fn, label_smoothing=label_smoothing
            )
        return loss_fn
    elif loss_name == "adaptively_smoothed_ranking_full":
        loss_fn = functools.partial(
            adaptive_label_smoothed_full_ranking_loss,
            label_smoothing_beta=label_smoothing_beta,
        )
        return loss_fn
    else:
        raise ValueError(f"Unsupported loss function: {loss_name}")