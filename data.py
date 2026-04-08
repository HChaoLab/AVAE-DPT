"""
data.py -- AnnData data interface for AVAE-DPT.
  - AnnDataset    : wraps an AnnData object as a PyTorch Dataset
  - make_dataloader : creates a DataLoader and auto-fills config.n_genes / config.n_patients
"""
import numpy as np
import scipy.sparse
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import anndata as ad
except ImportError:
    raise ImportError("anndata is required: pip install anndata")

from config import AVAEConfig


class AnnDataset(Dataset):
    """
    Wraps an AnnData object as a PyTorch Dataset.

    Expected AnnData structure:
        adata.X                   -- expression matrix (cells x genes)
                                     NB loss  : raw integer counts
                                     MSE loss : log-normalised values
        adata.obs[patient_col]    -- patient ID (string or categorical)
        adata.obs[stage_col]      -- stage label (0 = primary, 1 = metastasis; int or float)

    The sparse expression matrix is converted to dense float32 once in __init__
    and cached in memory, avoiding repeated conversion overhead in __getitem__.

    Parameters
    ----------
    adata:       AnnData object
    patient_col: obs column name for patient IDs, default "patient_id"
    stage_col:   obs column name for stage labels, default "stage"
    """

    def __init__(
        self,
        adata: "ad.AnnData",
        patient_col: str = "patient_id",
        stage_col: str = "stage",
    ):
        # Validate required columns
        for col in (patient_col, stage_col):
            if col not in adata.obs.columns:
                raise KeyError(
                    f"Column '{col}' not found in AnnData.obs. "
                    f"Available columns: {list(adata.obs.columns)}"
                )

        # Expression matrix -- convert sparse to dense float32 once
        if scipy.sparse.issparse(adata.X):
            self.X = np.asarray(adata.X.todense(), dtype=np.float32)
        else:
            self.X = np.array(adata.X, dtype=np.float32)
        # shape: (n_cells, n_genes)

        self.n_cells, self.n_genes = self.X.shape

        # Patient labels -> integer indices
        raw_patient = adata.obs[patient_col].astype(str).values
        unique_patients = sorted(set(raw_patient))
        self.patient_to_idx: dict = {p: i for i, p in enumerate(unique_patients)}
        self.n_patients: int = len(unique_patients)

        self.patient_ids = np.array(
            [self.patient_to_idx[p] for p in raw_patient], dtype=np.int64
        )

        # Stage labels
        self.stage_labels = np.array(adata.obs[stage_col].values, dtype=np.float32)

        # Library size per cell (total counts), shape (n_cells, 1), used to scale NB mean.
        # Clipped to at least 1 to prevent division-by-zero on all-zero rows.
        self.library_sizes = self.X.sum(axis=1, keepdims=True).astype(np.float32)
        self.library_sizes = np.clip(self.library_sizes, a_min=1.0, a_max=None)

    # -------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_cells

    def __getitem__(self, idx: int) -> dict:
        x = torch.from_numpy(self.X[idx])          # (n_genes,)

        patient_idx = int(self.patient_ids[idx])

        # One-hot encode patient label -> (n_patients,)
        batch_onehot = torch.zeros(self.n_patients, dtype=torch.float32)
        batch_onehot[patient_idx] = 1.0

        stage = torch.tensor(self.stage_labels[idx], dtype=torch.float32)

        # Library size shape (1,) for broadcasting in the NB loss
        library_size = torch.from_numpy(self.library_sizes[idx])  # (1,)

        return {
            "x":            x,
            "batch_onehot": batch_onehot,
            "patient_idx":  torch.tensor(patient_idx, dtype=torch.long),
            "stage":        stage,
            "library_size": library_size,
        }

    # -------------------------------------------------------------------
    def get_patient_mapping(self) -> dict:
        """Returns the patient-name -> integer-index mapping dict."""
        return dict(self.patient_to_idx)


# ---------------------------------------------------------------------------
# make_dataloader
# ---------------------------------------------------------------------------

def make_dataloader(
    adata: "ad.AnnData",
    config: AVAEConfig,
    patient_col: str = "patient_id",
    stage_col: str = "stage",
    shuffle: bool = True,
) -> DataLoader:
    """
    Creates a PyTorch DataLoader from an AnnData object and **mutates** config
    in-place to fill in n_genes and n_patients, which are required by AVAE.__init__.

    Parameters
    ----------
    adata:       AnnData object
    config:      AVAEConfig instance (modified in-place)
    patient_col: obs column name for patient IDs
    stage_col:   obs column name for stage labels
    shuffle:     whether to shuffle (True for training, False for inference)

    Returns
    -------
    DataLoader whose batches are dicts with keys: x, batch_onehot,
    patient_idx, stage, library_size.

    Usage
    -----
    >>> config = AVAEConfig()
    >>> loader = make_dataloader(adata, config, shuffle=True)
    >>> # config.n_genes and config.n_patients are now set automatically
    >>> model = AVAE(config)
    """
    dataset = AnnDataset(adata, patient_col=patient_col, stage_col=stage_col)

    # Auto-fill data dimensions into config
    config.n_genes    = dataset.n_genes
    config.n_patients = dataset.n_patients

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        # num_workers > 0 causes multiprocessing fork issues on Windows; keep at 0
        num_workers=0,
        pin_memory=(config.device != "cpu"),
        drop_last=False,
    )

    print(
        f"[make_dataloader] cells={dataset.n_cells}, "
        f"genes={dataset.n_genes}, "
        f"patients={dataset.n_patients}"
    )
    return loader
