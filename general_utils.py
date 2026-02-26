import numpy as np
import torch


def denormalize_positions(
    xy_norm: np.ndarray,
    min: np.ndarray,
    max: np.ndarray,
) -> np.ndarray:
    """
    Denormalize positions from [-1, 1] back to original scale.

    xy_norm: (N, pred_len, 2)
    min, max: shape (2,)

    Returns:
        xy_real: (N, pred_len, 2)
    """
    xy_real = (xy_norm + 1) * 0.5 * (max - min) + min
    return xy_real


def denormalize_positions_torch(
    xy_norm: torch.Tensor,
    min: np.ndarray,
    max: np.ndarray,
) -> torch.Tensor:
    """
    Denormalize positions from [-1, 1] back to original scale.

    xy_norm: (N, pred_len, 2)
    min, max: shape (2,)

    Returns:
        xy_real: (N, pred_len, 2)
    """
    
    if torch.is_tensor(min):
        min_t = min.to(device=xy_norm.device, dtype=xy_norm.dtype)
    else:
        min_t = torch.as_tensor(min, device=xy_norm.device, dtype=xy_norm.dtype)

    if torch.is_tensor(max):
        max_t = max.to(device=xy_norm.device, dtype=xy_norm.dtype)
    else:
        max_t = torch.as_tensor(max, device=xy_norm.device, dtype=xy_norm.dtype)
    
    # min_t = torch.tensor(min, device=xy_norm.device, dtype=xy_norm.dtype)  # (2,)
    # max_t = torch.tensor(max, device=xy_norm.device, dtype=xy_norm.dtype)
    xy_real = (xy_norm + 1) * 0.5 * (max_t - min_t) + min_t
    return xy_real


def denorm_xy_torch(xy_norm: torch.Tensor, min_xy, max_xy):
    """
    xy_norm: (..., 2) in [-1,1]
    min_xy, max_xy: array-like shape (2,)
    returns: (..., 2) in original units
    """
    if not torch.is_tensor(min_xy):
        min_xy = torch.tensor(min_xy, device=xy_norm.device, dtype=xy_norm.dtype)
    if not torch.is_tensor(max_xy):
        max_xy = torch.tensor(max_xy, device=xy_norm.device, dtype=xy_norm.dtype)

    return (xy_norm + 1.0) * 0.5 * (max_xy - min_xy) + min_xy