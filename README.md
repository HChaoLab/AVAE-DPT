# AVAE-DPT

**Adversarial Variational Autoencoder for Primary-to-Metastasis Pseudotime Inference**

AVAE-DPT is a PyTorch model for single-cell RNA-seq data that infers a continuous pseudotime score representing each cell's position along the primary-tumour-to-metastasis progression axis. It couples a VAE — whose decoder is conditioned on patient identity to disentangle batch effects from the latent code — with an adversarial patient discriminator connected via a Gradient Reversal Layer (GRL), while a supervised stage predictor guides the latent space to preserve the primary-to-metastasis biological signal.

---

## Motivation

Direct dimensionality reduction or batch correction on multi-patient scRNA-seq data often produces a latent space dominated by inter-patient variability rather than the biological process of interest. AVAE-DPT addresses this by:

1. **Adversarial batch removal** — A patient discriminator tries to predict patient origin from `z`. The GRL reverses its gradients back to the encoder, forcing `z` to be patient-invariant.
2. **Supervised stage guidance** — A stage predictor distinguishes primary (0) from metastatic (1) cells, with L1 sparsity encouraging only a few latent dimensions to drive this axis.
3. **Pseudotime scoring** — Raw stage logits are globally min-max normalised to produce a continuous, saturation-free pseudotime score in [0, 1].

---

## Model Architecture

```
  X (n_genes) ──► Encoder ──────────────► z (z_dim)
                     │                     │
                     │            ┌────────┼──────────────────────┐
                     │            │        │                      │
                     │            ▼        ▼                      ▼
                     │        Decoder   StagePredictor    PatientDiscriminator
                     │    (z ║ batch)   (L1 on W[:,j])        [GRL inside]
                     │        │              │                      │
                     │        ▼              ▼                      ▼
                     │   px_mean, px_r   stage logit          patient logits
                     │   (+ px_dropout   ─── L_stage          ─── L_disc
                     │    if ZINB)
                     │        │
                     │        ▼
                     │    L_recon (NB / ZINB / MSE)
                     │
                  (μ, log σ²)
                     │
                     ▼
                   L_KL
```

### Total Loss

```
L = L_recon  +  β · L_KL  +  γ(t) · L_stage  +  λ(t) · L_disc  +  α · L_L1_stage
```

| Term | Description |
|------|-------------|
| `L_recon` | Reconstruction loss — NB NLL, ZINB NLL, or MSE (see below) |
| `L_KL` | KL divergence of posterior vs. N(0, I) |
| `L_stage` | Binary cross-entropy, primary vs. metastasis |
| `L_disc` | Multi-class cross-entropy, patient classification (gradient reversed via GRL) |
| `L_L1_stage` | Element-wise L1 norm of the Stage Predictor's first linear layer weights |
| `γ(t)`, `λ(t)` | Time-varying weights controlled by the warm-up schedule |

---

## Reconstruction Loss Modes

Three modes are supported via `config.recon_loss`:

| Mode | `recon_loss` | Input data | Decoder outputs |
|------|-------------|------------|-----------------|
| **Negative Binomial** (default) | `"nb"` | Raw integer counts | `px_mean` (softmax proportion) + `px_r` (per-cell dispersion) |
| **Zero-Inflated NB** | `"zinb"` | Raw integer counts | `px_mean` + `px_r` + `px_dropout` (zero-inflation probability π) |
| **Mean Squared Error** | `"mse"` | Log-normalised values | `x_hat` (raw reconstruction) |

> **Why NB over ZINB for 10x data?** Academic consensus holds that technical zero-inflation is absent in modern 10x Genomics droplet data; using ZINB introduces unnecessary parameters. Use `"nb"` (default) unless you have strong biological reasons to model excess zeros.

The dispersion parameter `px_r` is output per-cell per-gene by a dedicated decoder head (no global `log_theta` parameter), making the model more expressive across diverse cell states.

---

## L1 Sparsity and Adaptive Dimension Decoupling

Rather than hardcoding which latent dimensions encode evolution, AVAE-DPT lets the model self-select:

- The full `z` (all `z_dim` dimensions) enters the Stage Predictor.
- An L1 penalty on the **first linear layer's weights** `W` (shape `[hidden_dim, z_dim]`) pushes most columns toward zero.
- After training, dimensions whose **Column L1 Norm** exceeds a threshold are the inferred evolutionary dimensions `z_evo`.

```python
# Inspect which z dimensions drive stage prediction
active_dims = model.get_active_stage_dims(threshold=1e-3)
# e.g. → Active dims: [2, 7, 14]  (3 / 20)
```

---

## Pseudotime vs. Stage Probability

Two distinct inference functions serve different purposes:

| Function | Method | Use case |
|----------|--------|----------|
| `get_pseudotime_score()` | logit → **global min-max normalisation** | Continuous trajectory visualisation |
| `get_stage_probability()` | logit → **sigmoid** | AUC / ROC / accuracy evaluation |

`get_pseudotime_score` avoids sigmoid saturation (which collapses many cells to 0 or 1) by operating in the unbounded logit space and normalising globally across the entire dataset.

---

## Training Schedule

AVAE-DPT uses a **three-phase unified warm-up** that controls both `γ(t)` and `λ(t)` with the same schedule, ensuring a clean VAE pre-training phase:

```
Epoch:   0 ──────────── delay_epochs ──────────── delay + ramp ──── max_epochs
                              │                         │
γ(t), λ(t):   0.0 ─────────  0.0 ──── linear ramp ──── 1.0 ──────── 1.0
```

| Phase | Condition | Effect |
|-------|-----------|--------|
| **Delay** | `epoch < delay_epochs` | `γ = λ = 0`; only reconstruction + KL active (pure VAE) |
| **Ramp** | `delay ≤ epoch < delay + ramp` | `γ` and `λ` increase linearly from 0 to target |
| **Full** | `epoch ≥ delay + ramp` | `γ` and `λ` at full configured values |

> This replaces the DANN sigmoid GRL schedule from v1. The trainer now explicitly computes and passes `gamma_eff` and `lambda_eff` each epoch rather than relying on an internal self-schedule inside the GRL layer.

KL annealing (`kl_warmup_epochs`) remains a separate counter.

---

## File Structure

```
AVAE-DPT/
├── config.py   # AVAEConfig dataclass — all hyperparameters
├── layers.py   # FCLayers, GradientReversal, compute_schedule_weight
├── model.py    # Encoder, Decoder (NB/ZINB/MSE), StagePredictor,
│               #   PatientDiscriminator, AVAE
│               #   → get_active_stage_dims()
│               #   → get_stage_logits() / get_stage_probability()
├── data.py     # AnnDataset and make_dataloader
└── train.py    # nb_loss, zinb_loss, kl_divergence, train_avae,
                #   get_pseudotime_score, get_stage_probability, smoke test
```

---

## Installation

```bash
pip install torch torchvision anndata scanpy scipy numpy
```

Python ≥ 3.9 and PyTorch ≥ 2.0 are recommended.

---

## Quick Start

### 1. Prepare AnnData

```python
import scanpy as sc

adata = sc.read_h5ad("tumor_data.h5ad")

# Select highly variable genes (recommended: 2000–5000)
sc.pp.highly_variable_genes(adata, n_top_genes=3000)
adata = adata[:, adata.var.highly_variable]

# Required obs columns:
#   "patient_id" -- string patient identifier
#   "stage"      -- integer label: 0 = primary, 1 = metastasis
#
# For NB / ZINB: adata.X must be RAW integer counts (do NOT log-normalise).
# For MSE:       adata.X should be log-normalised.
```

### 2. Configure and Train

```python
from config import AVAEConfig
from data import make_dataloader
from model import AVAE
from train import train_avae, get_pseudotime_score, get_stage_probability

config = AVAEConfig(
    max_epochs=200,
    z_dim=20,
    recon_loss="nb",         # "nb" (default) | "zinb" | "mse"
    beta=1.0,                # KL weight
    gamma=1.0,               # stage loss target weight
    adv_lambda_max=1.0,      # adversarial loss target weight
    alpha=1e-4,              # L1 sparsity weight on Stage Predictor
    delay_epochs=50,         # pure VAE warm-up (no stage / adversarial signal)
    ramp_epochs=50,          # linear ramp duration after delay
    kl_warmup_epochs=50,     # KL annealing duration
    device="cuda",
)

# make_dataloader auto-fills config.n_genes and config.n_patients
loader = make_dataloader(adata, config, shuffle=True)

model = AVAE(config)
model.summary()

history = train_avae(model, loader, config, verbose=True, print_every=10)
```

### 3. Extract Pseudotime

```python
# Continuous pseudotime via global logit min-max normalisation
pt = get_pseudotime_score(model, adata)
# → written to adata.obs["pseudotime"]

# Stage probability for classification metrics (AUC / accuracy)
prob = get_stage_probability(model, adata)
# → written to adata.obs["stage_prob"]

# Visualise
sc.pl.umap(adata, color=["pseudotime", "stage", "patient_id"])
```

### 4. Inspect Evolutionary Dimensions

```python
# After training, identify which z dimensions drive the primary→metastasis axis
active_dims = model.get_active_stage_dims(threshold=1e-3)
# Prints column L1 norms for all z_dim dimensions and returns active indices
```

---

## Smoke Test

Validates all three reconstruction loss modes end-to-end on synthetic data:

```bash
python train.py
```

Generates 200 synthetic cells across 3 patients, runs 6 epochs for each of `nb`, `zinb`, and `mse`, and checks that primary and metastatic cells receive distinct pseudotime scores.

---

## Hyperparameter Reference

### Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `z_dim` | 20 | Latent space dimensionality |
| `encoder_hidden_dims` | [256, 128] | Encoder hidden layer widths |
| `decoder_hidden_dims` | [128, 256] | Decoder hidden layer widths |
| `stage_hidden_dims` | [64] | Stage predictor hidden widths |
| `disc_hidden_dims` | [64] | Patient discriminator hidden widths |
| `dropout_rate` | 0.1 | Dropout probability in all FCLayers |
| `use_batch_norm` | True | BatchNorm1d in FCLayers |

### Loss

| Parameter | Default | Description |
|-----------|---------|-------------|
| `recon_loss` | `"nb"` | Reconstruction loss: `"nb"`, `"zinb"`, or `"mse"` |
| `beta` | 1.0 | KL divergence weight (final value after annealing) |
| `gamma` | 1.0 | Stage classification loss target weight |
| `adv_lambda_max` | 1.0 | Adversarial loss target weight (also sets GRL lambda ceiling) |
| `alpha` | 1e-4 | L1 penalty weight on Stage Predictor first layer |

### Schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delay_epochs` | 50 | Silent warm-up: `γ = λ = 0` (pure VAE reconstruction phase) |
| `ramp_epochs` | 50 | Epochs to linearly ramp `γ` and `λ` from 0 to target after delay |
| `kl_warmup_epochs` | 50 | Epochs to linearly ramp `β` from 0 to `beta` |

### Training

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 1e-3 | Adam learning rate |
| `batch_size` | 256 | Training mini-batch size |
| `max_epochs` | 200 | Total training epochs |
| `device` | `"cuda"` | `"cuda"` or `"cpu"` |

---

## Key Design Decisions

**Single optimizer.** A single Adam optimizer covers all parameters. The GRL converts the adversarial minimax game into a standard gradient-descent problem — no alternating optimizers needed.

**Decoder-output dispersion.** Each cell gets its own per-gene dispersion `px_r` from a dedicated decoder head, replacing the global per-gene `log_theta` parameter from v1. This allows dispersion to vary with cell state.

**Adaptive dimension decoupling via L1.** Rather than hardcoding which latent dimensions are "evolutionary", element-wise L1 on the Stage Predictor's first layer weight matrix naturally zeros out columns (z dimensions) that are irrelevant to stage prediction. `get_active_stage_dims()` reads back the inferred `z_evo` after training.

**Logit pseudotime, not sigmoid probability.** Sigmoid maps extreme logits to values indistinguishable from 0 or 1, collapsing trajectory resolution at the ends. Operating in logit space and applying global min-max normalisation preserves the full continuous distribution of cell positions.

**Unified warm-up schedule.** `γ(t)` and `λ(t)` share the same three-phase schedule (delay → linear ramp → full), replacing the DANN sigmoid self-schedule from v1. The trainer passes these weights explicitly each epoch, making the training dynamics fully transparent.

**KL annealing is separate.** The KL warm-up counter (`kl_warmup_epochs`) is independent of the stage/adversarial schedule, allowing reconstruction to stabilise before both regularisation pressures are applied.

---

## Reference

> Ganin, Y. et al. (2016). Domain-adversarial training of neural networks. *Journal of Machine Learning Research*, 17(59), 1–35.

> Lopez, R. et al. (2018). Deep generative modeling for single-cell transcriptomics. *Nature Methods*, 15, 1053–1058. (scVI)
