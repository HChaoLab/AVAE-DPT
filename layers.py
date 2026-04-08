"""
layers.py -- Reusable building blocks for AVAE-DPT.
  - FCLayers              : stacked fully-connected layers (Linear -> BN -> ReLU -> Dropout) x N
  - GradientReversalFunction : torch.autograd.Function implementing gradient reversal
  - GradientReversal      : nn.Module wrapper; lambda is injected by the trainer each step
  - compute_schedule_weight  : [CHANGED] unified epoch-level schedule for gamma and lambda,
                                replacing the DANN sigmoid schedule
"""
import math
from typing import List, Type

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# FCLayers
# ---------------------------------------------------------------------------

class FCLayers(nn.Module):
    """
    Configurable stacked fully-connected layers, aligned with scVI's FCLayers style.

    Per hidden layer:
        Linear(d_in, d_hidden)
        -> BatchNorm1d(d_hidden)  [optional]
        -> activation()
        -> Dropout(p)             [optional]

    Output layer:
        Linear(d_last_hidden, out_dim)
        (no BN / activation / Dropout -- the caller decides the output activation)

    Args:
        in_dim:         input dimensionality
        out_dim:        output dimensionality of the final Linear layer
        hidden_dims:    list of hidden layer widths, e.g. [256, 128]
                        empty list results in a single Linear(in_dim, out_dim)
        dropout_rate:   dropout probability; 0 disables dropout
        use_batch_norm: whether to insert BatchNorm1d after each hidden layer
        activation:     activation class, default nn.ReLU
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: List[int],
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        activation: Type[nn.Module] = nn.ReLU,
    ):
        super().__init__()
        dims = [in_dim] + list(hidden_dims) + [out_dim]
        layers: List[nn.Module] = []

        for i in range(len(dims) - 1):
            d_in, d_out = dims[i], dims[i + 1]
            is_last = (i == len(dims) - 2)

            layers.append(nn.Linear(d_in, d_out))

            if not is_last:
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(d_out, momentum=0.01, eps=1e-3))
                layers.append(activation())
                if dropout_rate > 0.0:
                    layers.append(nn.Dropout(p=dropout_rate))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer (GRL) from Ganin et al. (2016).

    Forward pass : identity mapping (y = x).
    Backward pass: multiply gradient by -lambda_val.

    lambda_val is stored as a plain Python float so it is never updated by
    the optimizer and never appears in the computation graph.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float) -> torch.Tensor:
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Second return value is for lambda_val (non-Tensor) -> None
        return -ctx.lambda_val * grad_output, None


class GradientReversal(nn.Module):
    """
    nn.Module wrapper around GradientReversalFunction.

    [CHANGED] lambda_val is now set exclusively by the trainer via set_lambda().
    The DANN sigmoid self-scheduling is removed; the trainer's compute_schedule_weight()
    controls both the GRL lambda and the adversarial loss weight uniformly.
    """

    def __init__(self, lambda_val: float = 0.0):
        super().__init__()
        self.lambda_val: float = lambda_val   # plain float, not nn.Parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, val: float) -> None:
        """Called by the trainer once per step with the current schedule value."""
        self.lambda_val = val

    def extra_repr(self) -> str:
        return f"lambda_val={self.lambda_val:.4f}"


# ---------------------------------------------------------------------------
# Unified schedule for gamma and lambda  [CHANGED: replaces compute_grl_lambda]
# ---------------------------------------------------------------------------

def compute_schedule_weight(
    epoch: int,
    delay_epochs: int,
    ramp_epochs: int,
) -> float:
    """
    Unified epoch-level weight schedule for the stage loss (gamma) and the
    adversarial loss / GRL lambda.  Replaces the DANN sigmoid schedule.

    Three phases:
      Phase 1  epoch < delay_epochs                     -> 0.0  (pure VAE phase)
      Phase 2  delay_epochs <= epoch < delay + ramp     -> linear ramp [0, 1]
      Phase 3  epoch >= delay_epochs + ramp_epochs      -> 1.0

    The trainer computes this weight and passes it explicitly to:
      - model.discriminator.grl.set_lambda(weight * config.adv_lambda_max)
      - multiplies stage loss and disc loss by the appropriate scaled weight

    Args:
        epoch:        current epoch index (0-based)
        delay_epochs: length of the silent warm-up phase (gamma = lambda = 0)
        ramp_epochs:  length of the linear ramp phase after the delay

    Returns:
        weight in [0.0, 1.0]
    """
    if epoch < delay_epochs:
        return 0.0
    elapsed = epoch - delay_epochs
    return min(1.0, elapsed / max(ramp_epochs, 1))
