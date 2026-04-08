"""
config.py -- AVAE-DPT hyperparameter configuration.
All model architecture and training parameters are centralised here.
n_genes and n_patients are filled in at runtime by make_dataloader().
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class AVAEConfig:
    # ------------------------------------------------------------------
    # Data dimensions (auto-filled by make_dataloader, do not set manually)
    # ------------------------------------------------------------------
    n_genes: int = 0          # AnnData.n_vars -- encoder input dimension
    n_patients: int = 0       # number of unique patients -- discriminator output classes

    # ------------------------------------------------------------------
    # Network architecture
    # ------------------------------------------------------------------
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    decoder_hidden_dims: List[int] = field(default_factory=lambda: [128, 256])
    stage_hidden_dims:   List[int] = field(default_factory=lambda: [64])
    disc_hidden_dims:    List[int] = field(default_factory=lambda: [64])
    z_dim: int = 20           # latent space dimensionality

    dropout_rate: float = 0.1
    use_batch_norm: bool = True

    # ------------------------------------------------------------------
    # Reconstruction loss
    # [CHANGED] Now supports three modes:
    #   "nb"   -- Negative Binomial NLL (default, recommended for raw counts)
    #   "zinb" -- Zero-Inflated NB (adds a per-gene dropout probability head)
    #   "mse"  -- Mean Squared Error  (for log-normalised input)
    # ------------------------------------------------------------------
    recon_loss: str = "nb"

    # ------------------------------------------------------------------
    # Loss weights
    # ------------------------------------------------------------------
    beta: float = 1.0           # KL divergence weight
    gamma: float = 1.0          # stage classification loss weight
    adv_lambda_max: float = 1.0  # adversarial (GRL) loss weight ceiling

    # [NEW] L1 regularisation coefficient on the Stage Predictor's first
    # linear layer weights -- promotes sparsity over z dimensions.
    alpha: float = 1e-4

    # ------------------------------------------------------------------
    # Warm-up / annealing schedule
    # [CHANGED] The DANN sigmoid schedule for GRL is REPLACED by a unified
    # epoch-level schedule controlled by delay_epochs and ramp_epochs.
    #
    # Phase 1  (epoch < delay_epochs)          : gamma = 0, lambda = 0
    # Phase 2  (delay_epochs <= epoch < delay + ramp): linear ramp to target
    # Phase 3  (epoch >= delay_epochs + ramp_epochs): gamma and lambda at full value
    #
    # KL annealing uses a separate kl_warmup_epochs counter (unchanged).
    # ------------------------------------------------------------------
    delay_epochs: int = 50      # pure VAE reconstruction phase (no stage / adv signal)
    ramp_epochs: int = 50       # epochs to linearly ramp gamma and lambda from 0 to target
    kl_warmup_epochs: int = 50  # KL annealing: linearly ramp beta from 0 to beta

    # ------------------------------------------------------------------
    # Training hyperparameters
    # ------------------------------------------------------------------
    lr: float = 1e-3
    batch_size: int = 256
    max_epochs: int = 200

    # ------------------------------------------------------------------
    # Runtime device (set before calling train_avae)
    # ------------------------------------------------------------------
    device: str = "cuda"     # "cuda" or "cpu"

    def __post_init__(self):
        assert self.recon_loss in ("nb", "zinb", "mse"), (
            f"recon_loss must be 'nb', 'zinb', or 'mse', got: {self.recon_loss}"
        )
        assert self.z_dim > 0, "z_dim must be a positive integer"
        assert self.kl_warmup_epochs >= 0, "kl_warmup_epochs must be non-negative"
        assert self.delay_epochs >= 0,     "delay_epochs must be non-negative"
        assert self.ramp_epochs >= 0,      "ramp_epochs must be non-negative"
        assert self.alpha >= 0,            "alpha (L1 weight) must be non-negative"
