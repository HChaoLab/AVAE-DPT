"""
model.py -- AVAE-DPT network modules.

Changes from v1:
  [1] Decoder now supports three recon_loss modes (nb / zinb / mse).
      For nb/zinb it outputs BOTH px_mean (softmax proportion) and px_r
      (per-cell-per-gene dispersion from a dedicated head).
      For zinb it additionally outputs px_dropout (dropout probability pi).
      The global log_theta parameter on AVAE is removed.

  [2] StagePredictor exposes a `first_linear` property pointing to the first
      Linear layer (weight shape: [hidden_dim, z_dim]).  This is used for
      the L1 sparsity regularisation and get_active_stage_dims().

  [3] AVAE adds:
      - get_active_stage_dims(threshold)  : Column-L1-Norm based active dim analysis
      - get_stage_logits(x)               : deterministic raw logits via mu
      - get_stage_probability(x)          : sigmoid probability for AUC / accuracy
      The old get_pseudotime() is removed; global min-max normalisation lives in train.py.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from config import AVAEConfig
from layers import FCLayers, GradientReversal


# ---------------------------------------------------------------------------
# 1. Encoder  (unchanged)
# ---------------------------------------------------------------------------

class Encoder(nn.Module):
    """
    Encodes gene expression matrix X into the parameters of the latent distribution.

    Input : x        (B, n_genes)
    Output:
        mu      (B, z_dim)  -- posterior mean
        log_var (B, z_dim)  -- posterior log-variance
        z       (B, z_dim)  -- reparameterised sample: z = mu + eps * exp(0.5 * log_var)
    """

    def __init__(
        self,
        n_genes: int,
        z_dim: int,
        hidden_dims,
        dropout_rate: float,
        use_batch_norm: bool,
    ):
        super().__init__()
        self.fc = FCLayers(
            in_dim=n_genes,
            out_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        self.fc_mu      = nn.Linear(hidden_dims[-1], z_dim)
        self.fc_log_var = nn.Linear(hidden_dims[-1], z_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        h = self.fc(x)
        mu      = self.fc_mu(h)
        log_var = torch.clamp(self.fc_log_var(h), min=-10.0, max=4.0)
        z       = self.reparameterize(mu, log_var)
        return mu, log_var, z


# ---------------------------------------------------------------------------
# 2. Decoder  [CHANGED: multi-mode output, dispersion is now a decoder head]
# ---------------------------------------------------------------------------

class Decoder(nn.Module):
    """
    Decodes (z || batch_onehot) into reconstructed expression parameters.

    batch_onehot is concatenated to z before the MLP, forcing the decoder to
    use patient identity for reconstruction so that z becomes batch-free.

    Output dict keys by recon_loss mode:
      "nb"   -> {"px_mean": (B,G),  "px_r": (B,G)}
      "zinb" -> {"px_mean": (B,G),  "px_r": (B,G),  "px_dropout": (B,G)}
      "mse"  -> {"x_hat":  (B,G)}

    [CHANGED] Three dedicated output heads replace the single self.out:
      mean_head       : softmax for proportion (nb/zinb) or raw output (mse)
      dispersion_head : exp-activated dispersion px_r = exp(clamp(h))  (nb/zinb)
      dropout_head    : sigmoid-activated dropout probability pi        (zinb only)

    [REMOVED] The global log_theta parameter is no longer on AVAE; dispersion
    is fully determined by the decoder per cell and gene.
    """

    def __init__(
        self,
        z_dim: int,
        n_patients: int,
        n_genes: int,
        hidden_dims,
        dropout_rate: float,
        use_batch_norm: bool,
        recon_loss: str = "nb",
    ):
        super().__init__()
        self.recon_loss = recon_loss
        in_dim = z_dim + n_patients

        # Shared backbone
        self.fc = FCLayers(
            in_dim=in_dim,
            out_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )

        # Mean output head (all modes)
        self.mean_head = nn.Linear(hidden_dims[-1], n_genes)

        # Dispersion head (nb and zinb modes)
        if recon_loss in ("nb", "zinb"):
            self.dispersion_head = nn.Linear(hidden_dims[-1], n_genes)

        # Dropout (zero-inflation) probability head (zinb mode only)
        if recon_loss == "zinb":
            self.dropout_head = nn.Linear(hidden_dims[-1], n_genes)

    def forward(
        self, z: torch.Tensor, batch_onehot: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        inp = torch.cat([z, batch_onehot], dim=-1)
        h   = self.fc(inp)

        if self.recon_loss == "nb":
            px_mean = F.softmax(self.mean_head(h), dim=-1)             # (B, n_genes)
            # Clamp log-dispersion for numerical stability; then exponentiate
            px_r    = torch.exp(
                self.dispersion_head(h).clamp(min=-10.0, max=4.0)
            )                                                           # (B, n_genes)
            return {"px_mean": px_mean, "px_r": px_r}

        elif self.recon_loss == "zinb":
            px_mean    = F.softmax(self.mean_head(h), dim=-1)
            px_r       = torch.exp(
                self.dispersion_head(h).clamp(min=-10.0, max=4.0)
            )
            px_dropout = torch.sigmoid(self.dropout_head(h))           # (B, n_genes), pi
            return {"px_mean": px_mean, "px_r": px_r, "px_dropout": px_dropout}

        else:  # mse
            # Raw output -- suitable for log-normalised data
            return {"x_hat": self.mean_head(h)}


# ---------------------------------------------------------------------------
# 3. StagePredictor  [CHANGED: adds first_linear property for L1 / active dims]
# ---------------------------------------------------------------------------

class StagePredictor(nn.Module):
    """
    Predicts the developmental stage of each cell from z (primary=0 / metastasis=1).

    Returns a raw logit (no sigmoid).  Consumers choose:
      - BCEWithLogitsLoss in training
      - torch.sigmoid()      for probability (AUC / accuracy)
      - min-max on logit     for continuous pseudotime (train.py)

    [NEW] first_linear property exposes the first Linear layer whose weights
    are regularised with L1 and inspected by get_active_stage_dims().

    Input : z     (B, z_dim)
    Output: logit (B, 1)
    """

    def __init__(
        self,
        z_dim: int,
        hidden_dims,
        dropout_rate: float,
        use_batch_norm: bool,
    ):
        super().__init__()
        self.fc  = FCLayers(
            in_dim=z_dim,
            out_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        self.out = nn.Linear(hidden_dims[-1], 1)

    @property
    def first_linear(self) -> nn.Linear:
        """
        The first Linear layer in fc.network.
        Weight shape: (first_hidden_dim, z_dim).

        Column j of this matrix represents z_j's contribution to all hidden
        units.  The Column L1 Norm of column j = sum_i |W[i, j]| measures
        how strongly dimension j influences stage prediction.
        """
        for m in self.fc.network:
            if isinstance(m, nn.Linear):
                return m
        raise RuntimeError(
            "No Linear layer found in StagePredictor.fc.network. "
            "Check that stage_hidden_dims is non-empty."
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.out(self.fc(z))


# ---------------------------------------------------------------------------
# 4. PatientDiscriminator  (GRL lambda set externally by trainer)
# ---------------------------------------------------------------------------

class PatientDiscriminator(nn.Module):
    """
    Classifies patient origin from z.  The embedded GRL reverses gradients
    to the encoder, training it to produce patient-invariant representations.

    Forward: z -> GRL -> FCLayers -> Linear -> logits

    [CHANGED] GRL lambda is set by compute_schedule_weight() in the trainer,
    replacing the old DANN sigmoid self-schedule.

    Input : z      (B, z_dim)
    Output: logits (B, n_patients)
    """

    def __init__(
        self,
        z_dim: int,
        n_patients: int,
        hidden_dims,
        dropout_rate: float,
        use_batch_norm: bool,
    ):
        super().__init__()
        self.grl = GradientReversal(lambda_val=0.0)   # starts silent
        self.fc  = FCLayers(
            in_dim=z_dim,
            out_dim=hidden_dims[-1],
            hidden_dims=hidden_dims[:-1],
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
        )
        self.out = nn.Linear(hidden_dims[-1], n_patients)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z_rev = self.grl(z)     # identity forward; gradient negated backward
        return self.out(self.fc(z_rev))


# ---------------------------------------------------------------------------
# 5. AVAE -- top-level model  [CHANGED: no log_theta; new inference methods]
# ---------------------------------------------------------------------------

class AVAE(nn.Module):
    """
    Adversarial Variational AutoEncoder for Pseudotime inference (AVAE-DPT).

    Changes from v1:
      - No global log_theta parameter; dispersion is now a Decoder output head.
      - forward() returns a flat dict that spreads all Decoder output keys.
      - get_active_stage_dims() uses Column L1 Norm to find evolutionary z dims.
      - get_stage_logits() / get_stage_probability() replace get_pseudotime().
        Global min-max normalisation for final pseudotime lives in train.py.
    """

    def __init__(self, config: AVAEConfig):
        super().__init__()
        self.config = config

        self.encoder = Encoder(
            n_genes=config.n_genes,
            z_dim=config.z_dim,
            hidden_dims=config.encoder_hidden_dims,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )
        self.decoder = Decoder(
            z_dim=config.z_dim,
            n_patients=config.n_patients,
            n_genes=config.n_genes,
            hidden_dims=config.decoder_hidden_dims,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
            recon_loss=config.recon_loss,   # "nb" / "zinb" / "mse"
        )
        self.stage_predictor = StagePredictor(
            z_dim=config.z_dim,
            hidden_dims=config.stage_hidden_dims,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )
        self.discriminator = PatientDiscriminator(
            z_dim=config.z_dim,
            n_patients=config.n_patients,
            hidden_dims=config.disc_hidden_dims,
            dropout_rate=config.dropout_rate,
            use_batch_norm=config.use_batch_norm,
        )
        # [REMOVED] self.log_theta -- dispersion is now a Decoder head

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor, batch_onehot: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass; returns a flat dict for loss computation.

        Common keys (all modes):
            "mu"          (B, z_dim)
            "log_var"     (B, z_dim)
            "z"           (B, z_dim)
            "stage_logit" (B, 1)
            "disc_logit"  (B, n_patients)

        Additional keys by recon_loss:
            nb   : "px_mean" (B,G), "px_r" (B,G)
            zinb : "px_mean" (B,G), "px_r" (B,G), "px_dropout" (B,G)
            mse  : "x_hat"   (B,G)
        """
        mu, log_var, z = self.encoder(x)
        dec_out        = self.decoder(z, batch_onehot)   # dict, keys vary by mode
        stage_logit    = self.stage_predictor(z)
        disc_logit     = self.discriminator(z)

        return {
            "mu":          mu,
            "log_var":     log_var,
            "z":           z,
            **dec_out,                  # spreads px_mean/px_r/px_dropout or x_hat
            "stage_logit": stage_logit,
            "disc_logit":  disc_logit,
        }

    # ------------------------------------------------------------------
    # Inference helpers  [NEW / CHANGED]
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_stage_logits(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns raw (pre-sigmoid) stage logits for every cell using deterministic mu.

        These logits are the input to global min-max normalisation in
        get_pseudotime_score() (train.py).  Using mu rather than sampled z
        ensures reproducible output across calls.

        Args:
            x: expression matrix  (n_cells, n_genes), float32, on the correct device
        Returns:
            logits: (n_cells,) float32
        """
        was_training = self.training
        self.eval()

        mu, _, _    = self.encoder(x)
        logit       = self.stage_predictor(mu).squeeze(-1)   # (n_cells,)

        if was_training:
            self.train()
        return logit

    @torch.no_grad()
    def get_stage_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns sigmoid-activated stage probabilities.

        Intended for classification metrics (Accuracy / AUC / ROC), NOT for
        pseudotime scoring (sigmoid saturates at extremes and produces poor
        trajectory resolution).

        Args:
            x: expression matrix  (n_cells, n_genes)
        Returns:
            probs: (n_cells,) float32 in [0, 1]
        """
        return torch.sigmoid(self.get_stage_logits(x))

    # ------------------------------------------------------------------
    # Latent space analysis  [NEW]
    # ------------------------------------------------------------------

    def get_active_stage_dims(self, threshold: float = 1e-3):
        """
        Identifies which z dimensions actively drive stage prediction.

        Algorithm (Column L1 Norm):
          1. Extract weight matrix W of the Stage Predictor's first Linear layer.
             Shape: (first_hidden_dim, z_dim).
          2. For each z dimension j (column), compute:
                col_l1[j] = sum_i |W[i, j]|
          3. Return indices j where col_l1[j] > threshold.

        After L1 regularisation (alpha * W.abs().sum() in the loss), many
        columns converge to near-zero, leaving only the dimensions that are
        truly relevant to the primary-to-metastasis axis.

        Args:
            threshold: Column L1 Norm cutoff; default 1e-3.

        Returns:
            active_indices: 1-D LongTensor of active z dimension indices.
        """
        W = self.stage_predictor.first_linear.weight.detach()   # (hidden_dim, z_dim)
        col_l1_norms = W.abs().sum(dim=0)                        # (z_dim,)

        active_mask    = col_l1_norms > threshold
        active_indices = active_mask.nonzero(as_tuple=False).squeeze(-1)

        print(f"[get_active_stage_dims] threshold = {threshold}")
        print(f"  Active dims : {active_indices.tolist()} "
              f"({active_mask.sum().item()} / {self.config.z_dim})")
        print(f"  Col L1 norms: {[f'{v:.4f}' for v in col_l1_norms.tolist()]}")
        return active_indices

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Prints a brief parameter count summary."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"AVAE-DPT [{self.config.recon_loss.upper()}]  "
              f"total params: {total:,}  trainable: {trainable:,}")
        for name, module in [
            ("Encoder",        self.encoder),
            ("Decoder",        self.decoder),
            ("StagePredictor", self.stage_predictor),
            ("Discriminator",  self.discriminator),
        ]:
            n = sum(p.numel() for p in module.parameters())
            print(f"  {name:<20s}: {n:>10,} params")
