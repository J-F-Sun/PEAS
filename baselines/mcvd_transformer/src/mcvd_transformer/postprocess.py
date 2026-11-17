from __future__ import annotations
import torch

def _causal_moving_average(values: torch.Tensor, window: int) -> torch.Tensor:
    if window < 1:
        raise ValueError("Moving-average window must be positive.")
    padded = torch.nn.functional.pad(values, (window - 1, 0))
    return torch.stack([padded[:, index : index + values.shape[1]] for index in range(window)]).mean(dim=0)


def postprocess_mcvd_output(
    shape: torch.Tensor,
    maximum: torch.Tensor,
    *,
    zero_padding: int,
    moving_average_window: int,
    epsilon: float = 1.0e-12,
) -> torch.Tensor:
    """Official MCvD-style monotonic curve reconstruction."""

    if shape.ndim != 2 or maximum.ndim != 2 or maximum.shape[1] != 1:
        raise ValueError("Expected shape=(batch,time) and maximum=(batch,1).")
    if zero_padding < 0 or zero_padding >= shape.shape[1] - 1:
        raise ValueError("zero_padding must leave at least two time samples.")
    differences = _causal_moving_average(torch.relu(shape[:, 1:] - shape[:, :-1]), moving_average_window)
    reconstructed = torch.cat([torch.zeros_like(shape[:, :1]), torch.cumsum(differences, dim=1)], dim=1) + shape[:, :1]
    reconstructed = reconstructed[:, zero_padding:]
    return reconstructed / reconstructed[:, -1:].clamp_min(epsilon) * maximum

