"""Prepare a tutorial-sized subset of Replogle-E-K562 for causarray batch demo.

Selects the top-200 most-abundant perturbations plus 2 000 randomly chosen
control cells and writes the result to ``replogle_subset.h5ad`` in the same
directory.  Run once from the project root:

    python docs/source/tutorial/replogle/prep_tutorial_data.py
"""
import numpy as np
import scanpy as sc
from pathlib import Path

# TODO: Set DATA_PATH to the location of your downloaded Replogle-E-K562 h5ad file.
DATA_PATH = Path("/path/to/Replogle-E-k562.h5ad")
OUT_DIR   = Path(__file__).parent
OUT_PATH  = OUT_DIR / "replogle_subset.h5ad"

CTRL_LABEL = "non-targeting"
PERT_COL   = "gene"
N_TOP_PERT = 200
N_CTRL     = 2000
RANDOM_STATE = 0

print(f"Reading {DATA_PATH} ...")
adata = sc.read_h5ad(DATA_PATH)
print(f"  Full dataset: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")

vc = adata.obs[PERT_COL].value_counts()
top_perts = vc[vc.index != CTRL_LABEL].head(N_TOP_PERT).index.tolist()
print(f"  Top {N_TOP_PERT} perturbations selected "
      f"(min {vc[top_perts].min()} – max {vc[top_perts].max()} cells)")

pert_mask = adata.obs[PERT_COL].isin(top_perts)

ctrl_idx = np.where(adata.obs[PERT_COL] == CTRL_LABEL)[0]
rng = np.random.default_rng(RANDOM_STATE)
ctrl_sel = np.sort(rng.choice(ctrl_idx, size=min(N_CTRL, len(ctrl_idx)), replace=False))
ctrl_mask = np.zeros(len(adata), dtype=bool)
ctrl_mask[ctrl_sel] = True

keep_mask = pert_mask | ctrl_mask
adata_sub = adata[keep_mask].copy()
adata_sub.uns["ctrl_label"] = CTRL_LABEL
adata_sub.uns["pert_col"]   = PERT_COL

n_ctrl_kept = ctrl_mask.sum()
n_pert_kept = pert_mask.sum()
print(f"  Subset: {n_pert_kept:,} pert cells + {n_ctrl_kept:,} ctrl cells "
      f"= {len(adata_sub):,} total")
print(f"  Writing {OUT_PATH} ...")
adata_sub.write_h5ad(OUT_PATH)
print("Done.")
