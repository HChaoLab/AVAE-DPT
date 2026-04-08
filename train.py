"""
train.py -- Loss functions, training loop, and pseudotime extraction.

Changes from v1:
  [1] nb_loss: theta now comes from the Decoder output (out["px_r"]) instead
      of the removed global AVAE.log_theta parameter.

  [2] zinb_loss: new function for Zero-Inflated NB loss using the logaddexp
      trick for numerical stability.

  [3] train_avae: the DANN sigmoid GRL schedule is replaced by the unified
      compute_schedule_weight() schedule.  gamma and lambda weights are
      computed once per epoch and passed explicitly to every loss term and to
      model.discriminator.grl.set_lambda().

  [4] Pseudotime:
      - get_pseudotime_score(): collects ALL logits across the dataset first,
        then applies global min-max normalisation -- avoids batch-boundary
        artefacts.
      - get_stage_probability(): returns sigmoid probabilities for AUC/accuracy.
      - The old get_pseudotime() (sigmoid-based) is removed.
"""
from typing import Optional

import numpy as np
import scipy.sparse
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import AVAEConfig
from layers import compute_schedule_weight
from model import AVAE


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def nb_loss(
    x: torch.Tensor,
    px_mean: torch.Tensor,
    px_r: torch.Tensor,
    library_size: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Negative Binomial reconstruction loss (mean negative log-likelihood).

    Parameterisation:
        mu = library_size * px_mean   (predicted count mean, per cell per gene)
        r  = px_r                     (dispersion, now from the Decoder head)

    log p(x | mu, r) =
        lgamma(x + r) - lgamma(r) - lgamma(x + 1)
        + r * [log r - log(r + mu)]
        + x * [log mu - log(r + mu)]

    [CHANGED] px_r is the Decoder's dispersion head output (B, n_genes),
    replacing the old global AVAE.log_theta.

    Args:
        x:            observed raw counts          (B, n_genes)
        px_mean:      decoder softmax proportions  (B, n_genes)
        px_r:         decoder dispersion output    (B, n_genes), already exp-activated
        library_size: total counts per cell        (B, 1)
        eps:          numerical stability floor

    Returns:
        scalar: mean negative log-likelihood over batch and genes
    """
    mu = (library_size * px_mean).clamp(min=eps)   # (B, n_genes)
    r  = px_r.clamp(min=eps)                        # (B, n_genes)

    log_r   = torch.log(r   + eps)
    log_mu  = torch.log(mu  + eps)
    log_rmu = torch.log(r + mu + eps)

    term1 = torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0)
    term2 = r * (log_r - log_rmu)
    term3 = x * (log_mu - log_rmu)

    return -(term1 + term2 + term3).mean()


def zinb_loss(
    x: torch.Tensor,
    px_mean: torch.Tensor,
    px_r: torch.Tensor,
    px_dropout: torch.Tensor,
    library_size: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Zero-Inflated Negative Binomial reconstruction loss.

    p(x | mu, r, pi) = pi * delta(x=0)  +  (1 - pi) * NB(x | mu, r)

    Case x == 0:
        log p = logaddexp( log(pi),  log(1-pi) + log NB(0|mu,r) )
              where log NB(0|mu,r) = r * [log r - log(r+mu)]

    Case x > 0:
        log p = log(1-pi) + log NB(x|mu,r)

    logaddexp is used for the x=0 branch to avoid exp() overflow / underflow.

    [NEW] This function is selected when config.recon_loss == "zinb".

    Args:
        x:           observed raw counts              (B, n_genes)
        px_mean:     decoder softmax proportions      (B, n_genes)
        px_r:        decoder dispersion output        (B, n_genes), exp-activated
        px_dropout:  decoder dropout probability pi   (B, n_genes), sigmoid-activated
        library_size: total counts per cell           (B, 1)
        eps:         numerical stability floor

    Returns:
        scalar: mean negative log-likelihood
    """
    mu = (library_size * px_mean).clamp(min=eps)
    r  = px_r.clamp(min=eps)
    pi = px_dropout.clamp(min=eps, max=1.0 - eps)

    log_r   = torch.log(r   + eps)
    log_mu  = torch.log(mu  + eps)
    log_rmu = torch.log(r + mu + eps)

    # log NB(0 | mu, r) = r * [log r - log(r + mu)]
    log_nb_zero = r * (log_r - log_rmu)

    # log NB(x | mu, r)  -- general
    log_nb_x = (
        torch.lgamma(x + r) - torch.lgamma(r) - torch.lgamma(x + 1.0)
        + r * (log_r - log_rmu)
        + x * (log_mu - log_rmu)
    )

    # x == 0 branch: log[ pi + (1-pi)*NB(0) ] via logaddexp for stability
    log_p_zero = torch.logaddexp(
        torch.log(pi),
        torch.log1p(-pi) + log_nb_zero,
    )

    # x > 0 branch: log[ (1-pi)*NB(x) ]
    log_p_nonzero = torch.log1p(-pi) + log_nb_x

    # Select branch based on whether the observed count is zero
    is_zero  = (x < 0.5).float()   # integer counts: 0 -> 1.0, >0 -> 0.0
    log_prob = is_zero * log_p_zero + (1.0 - is_zero) * log_p_nonzero

    return -log_prob.mean()


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    Closed-form KL divergence: KL[q(z|x) || N(0,I)].
    = -0.5 * mean(1 + log_var - mu^2 - exp(log_var))
    """
    return (-0.5 * (1.0 + log_var - mu.pow(2) - log_var.exp())).mean()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_avae(
    model: AVAE,
    dataloader: torch.utils.data.DataLoader,
    config: AVAEConfig,
    verbose: bool = True,
    print_every: int = 1,
) -> list:
    """
    Full AVAE-DPT training loop.

    Schedule design  [CHANGED from v1]:
      The DANN sigmoid GRL schedule is fully replaced by compute_schedule_weight().

      Per epoch:
        kl_weight      = KL annealing  (linear ramp over kl_warmup_epochs)
        schedule_w     = compute_schedule_weight(epoch, delay_epochs, ramp_epochs)
        gamma_eff      = config.gamma        * schedule_w   (stage loss weight)
        lambda_eff     = config.adv_lambda_max * schedule_w  (adversarial weight)

      model.discriminator.grl.set_lambda(lambda_eff) is called once per epoch
      so that GRL gradient reversal strength matches the loss weight exactly.

      During the delay phase (schedule_w == 0):
        - stage loss and adversarial loss contribute ZERO to the total loss
        - GRL lambda is 0 so no adversarial gradients reach the encoder
        - Only reconstruction + KL are active --> pure VAE warm-up

    L1 regularisation  [NEW]:
      l1_stage = stage_predictor.first_linear.weight.abs().sum()
      total loss += config.alpha * l1_stage

    Args:
        model:       initialised AVAE instance
        dataloader:  DataLoader from make_dataloader
        config:      AVAEConfig
        verbose:     print epoch summaries
        print_every: print interval

    Returns:
        history: List[dict] with per-epoch loss records
    """
    device = torch.device("cpu")
    if config.device == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            print("[train_avae] Warning: CUDA not available, falling back to CPU.")

    model.to(device)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    history: list = []

    for epoch in range(config.max_epochs):

        # ── KL annealing weight (separate from stage/adv schedule) ──────
        if config.kl_warmup_epochs > 0:
            kl_weight = min(1.0, epoch / config.kl_warmup_epochs) * config.beta
        else:
            kl_weight = config.beta

        # ── Unified schedule weight for gamma and lambda  [CHANGED] ──────
        # Single scalar in [0, 1] that controls BOTH stage and adversarial
        # losses.  Strictly 0 during delay phase.
        schedule_w   = compute_schedule_weight(
            epoch, config.delay_epochs, config.ramp_epochs
        )
        gamma_eff    = config.gamma          * schedule_w
        lambda_eff   = config.adv_lambda_max * schedule_w

        # Propagate lambda to GRL -- one explicit set call per epoch.
        # During delay phase lambda_eff == 0, so GRL produces zero adversarial
        # gradients even though disc_logit is still computed (for monitoring).
        model.discriminator.grl.set_lambda(lambda_eff)

        ep_total = ep_recon = ep_kl = ep_stage = ep_disc = ep_l1 = 0.0
        n_batches = 0

        for batch in dataloader:
            x            = batch["x"].to(device)             # (B, n_genes)
            batch_onehot = batch["batch_onehot"].to(device)   # (B, n_patients)
            patient_idx  = batch["patient_idx"].to(device)    # (B,) long
            stage        = batch["stage"].to(device)           # (B,) float
            library_size = batch["library_size"].to(device)    # (B, 1)

            # Forward pass
            out = model(x, batch_onehot)

            # ── Reconstruction loss (mode-dependent) ─────────────────────
            if config.recon_loss == "nb":
                l_recon = nb_loss(x, out["px_mean"], out["px_r"], library_size)
            elif config.recon_loss == "zinb":
                l_recon = zinb_loss(
                    x, out["px_mean"], out["px_r"], out["px_dropout"], library_size
                )
            else:  # mse
                l_recon = F.mse_loss(out["x_hat"], x)

            # ── KL divergence ─────────────────────────────────────────────
            l_kl = kl_divergence(out["mu"], out["log_var"])

            # ── Stage classification loss (scaled by gamma_eff) ───────────
            l_stage = F.binary_cross_entropy_with_logits(
                out["stage_logit"].squeeze(-1), stage
            )

            # ── Adversarial discriminator loss (scaled by lambda_eff) ─────
            # Positive coefficient here -- GRL in PatientDiscriminator
            # ensures the encoder receives reversed gradients automatically.
            l_disc = F.cross_entropy(out["disc_logit"], patient_idx)

            # ── L1 sparsity on Stage Predictor first layer  [NEW] ─────────
            # Encourages only a few z dimensions to drive stage prediction.
            l1_stage = model.stage_predictor.first_linear.weight.abs().sum()

            # ── Total loss ────────────────────────────────────────────────
            loss = (
                l_recon
                + kl_weight  * l_kl
                + gamma_eff  * l_stage
                + lambda_eff * l_disc
                + config.alpha * l1_stage
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            ep_total += loss.item()
            ep_recon += l_recon.item()
            ep_kl    += l_kl.item()
            ep_stage += l_stage.item()
            ep_disc  += l_disc.item()
            ep_l1    += l1_stage.item()
            n_batches += 1

        record = {
            "epoch":      epoch,
            "total":      ep_total  / n_batches,
            "recon":      ep_recon  / n_batches,
            "kl":         ep_kl     / n_batches,
            "stage":      ep_stage  / n_batches,
            "disc":       ep_disc   / n_batches,
            "l1_stage":   ep_l1     / n_batches,
            "kl_weight":  kl_weight,
            "gamma_eff":  gamma_eff,
            "lambda_eff": lambda_eff,
        }
        history.append(record)

        if verbose and (epoch % print_every == 0 or epoch == config.max_epochs - 1):
            print(
                f"Epoch {epoch:>4d}/{config.max_epochs} | "
                f"total={record['total']:.4f}  "
                f"recon={record['recon']:.4f}  "
                f"kl={record['kl']:.4f}  "
                f"stage={record['stage']:.4f}  "
                f"disc={record['disc']:.4f}  "
                f"l1={record['l1_stage']:.4f}  "
                f"beta={kl_weight:.3f}  "
                f"gamma={gamma_eff:.3f}  "
                f"lambda={lambda_eff:.3f}"
            )

    return history


# ---------------------------------------------------------------------------
# Inference: pseudotime score  [CHANGED from get_pseudotime]
# ---------------------------------------------------------------------------

def get_pseudotime_score(
    model: AVAE,
    adata,
    batch_size: int = 512,
    device: Optional[str] = None,
    obs_key: str = "pseudotime",
) -> np.ndarray:
    """
    Global min-max normalised pseudotime score derived from stage logits.

    Pipeline:
        X -> Encoder(mu) -> StagePredictor -> logit
        -> global min-max normalisation -> pseudotime in [0, 1]

    Key difference from sigmoid probability:
      Logit values are in an unbounded linear space.  Global min-max
      normalisation maps the full range of logits to [0, 1] without
      saturation, preserving fine-grained trajectory structure.

    Logits for ALL cells are collected before normalisation to ensure the
    scale is truly global (not batch-dependent).

    [CHANGED] Replaces the old get_pseudotime() (sigmoid-based).

    Args:
        model:      trained AVAE instance
        adata:      AnnData with the same gene dimension as training data
        batch_size: cells per forward pass
        device:     inference device; None -> model's current device
        obs_key:    adata.obs column name to write, default "pseudotime"

    Returns:
        pseudotime: np.ndarray (n_cells,), range [0, 1]
    """
    if device is None:
        try:
            device_obj = next(model.parameters()).device
        except StopIteration:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    X = _to_dense_float32(adata.X)

    model.eval()
    all_logits: list = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[start:start + batch_size], device=device_obj)
            logits  = model.get_stage_logits(x_batch)   # (batch,)
            all_logits.append(logits.cpu())

    logits_all = torch.cat(all_logits, dim=0)   # (n_cells,)

    # Global min-max normalisation
    lo, hi = logits_all.min(), logits_all.max()
    if hi > lo:
        pseudotime = ((logits_all - lo) / (hi - lo)).numpy()
    else:
        pseudotime = np.zeros(len(logits_all), dtype=np.float32)

    adata.obs[obs_key] = pseudotime
    print(
        f"[get_pseudotime_score] Written to adata.obs['{obs_key}']  "
        f"min={pseudotime.min():.4f}  max={pseudotime.max():.4f}  "
        f"mean={pseudotime.mean():.4f}"
    )
    return pseudotime


# ---------------------------------------------------------------------------
# Inference: stage probability  [NEW]
# ---------------------------------------------------------------------------

def get_stage_probability(
    model: AVAE,
    adata,
    batch_size: int = 512,
    device: Optional[str] = None,
    obs_key: str = "stage_prob",
) -> np.ndarray:
    """
    Sigmoid-activated stage probabilities for classification evaluation.

    Use this for AUC / ROC / accuracy metrics.
    Do NOT use this as the pseudotime score (sigmoid saturates at extremes).

    Args:
        model:      trained AVAE instance
        adata:      AnnData
        batch_size: cells per forward pass
        device:     inference device
        obs_key:    adata.obs column name to write, default "stage_prob"

    Returns:
        probs: np.ndarray (n_cells,), range [0, 1]
    """
    if device is None:
        try:
            device_obj = next(model.parameters()).device
        except StopIteration:
            device_obj = torch.device("cpu")
    else:
        device_obj = torch.device(device)

    X = _to_dense_float32(adata.X)

    model.eval()
    all_probs: list = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            x_batch = torch.tensor(X[start:start + batch_size], device=device_obj)
            probs   = model.get_stage_probability(x_batch)
            all_probs.append(probs.cpu())

    probs_all = torch.cat(all_probs, dim=0).numpy()

    adata.obs[obs_key] = probs_all
    print(
        f"[get_stage_probability] Written to adata.obs['{obs_key}']  "
        f"mean(primary)={probs_all.mean():.4f}"
    )
    return probs_all


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _to_dense_float32(X) -> np.ndarray:
    """Converts sparse or dense matrix to float32 numpy array."""
    if scipy.sparse.issparse(X):
        return np.asarray(X.todense(), dtype=np.float32)
    return np.array(X, dtype=np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def _smoke_test():
    """
    End-to-end validation on synthetic data.  Run with:  python train.py
    Tests all three recon_loss modes sequentially.
    """
    import anndata as ad
    from data import make_dataloader

    np.random.seed(42)
    torch.manual_seed(42)

    n_cells, n_genes, n_patients = 200, 500, 3

    X = np.random.negative_binomial(
        n=5, p=0.5, size=(n_cells, n_genes)
    ).astype(np.float32)

    patient_ids = np.array([f"P{i % n_patients}" for i in range(n_cells)])
    stage       = np.array(
        [0] * (n_cells // 2) + [1] * (n_cells // 2), dtype=np.float32
    )

    adata = ad.AnnData(X=X)
    adata.obs["patient_id"] = patient_ids
    adata.obs["stage"]      = stage

    for mode in ("nb", "zinb", "mse"):
        print(f"\n{'='*60}")
        print(f"  Smoke test: recon_loss = {mode}")
        print(f"{'='*60}")

        config = AVAEConfig(
            max_epochs=6,
            batch_size=64,
            z_dim=10,
            encoder_hidden_dims=[64, 32],
            decoder_hidden_dims=[32, 64],
            stage_hidden_dims=[32],
            disc_hidden_dims=[32],
            recon_loss=mode,
            alpha=1e-4,
            delay_epochs=2,
            ramp_epochs=2,
            kl_warmup_epochs=2,
            device="cpu",
        )

        loader = make_dataloader(adata, config, shuffle=True)
        model  = AVAE(config)
        model.summary()

        history = train_avae(model, loader, config, verbose=True, print_every=1)
        print(f"Final total loss: {history[-1]['total']:.4f}")

        pt = get_pseudotime_score(model, adata, batch_size=64, device="cpu")
        print(f"Primary    cells mean pseudotime : {pt[:n_cells//2].mean():.4f}")
        print(f"Metastasis cells mean pseudotime : {pt[n_cells//2:].mean():.4f}")

        model.get_active_stage_dims(threshold=1e-3)

    print("\nSmoke test passed for all three recon_loss modes.")


if __name__ == "__main__":
    _smoke_test()
