# Changelog

## [0.0.6] - 2026-05-31

### Added

- **Batch-wise fitting API** (`causarray/gcate.py`, `causarray/DR_learner.py`, `causarray/utils.py`)
  - `fit_gcate_batch(Y, X, A, r, batch_size=10, max_cells=2000, n_ctrl=2000, ...)`:
    Fits GCATE independently on batches of `batch_size` perturbations.  A fixed
    subsample of `n_ctrl` control cells is shared across all batches, dispersion
    is pre-estimated once on the control pool, and pert cells are capped at
    `max_cells=2000` (ctrl added on top).  Accepts `skip_batches` to resume
    interrupted runs.  Per-batch wall time and ETA reported when `verbose=True`.
  - `gcate_lfc_batch(Y, X, A, r, batch_size=10, max_cells=2000, n_ctrl=2000,
    cache_path=None, ...)`:
    End-to-end batch pipeline — runs GCATE *and* LFC per batch, freeing all
    large intermediate arrays (`res_1`, `res_2`, `Y_hat`, `pi_hat`) after each
    batch.  `cache_path` enables HDF5 disk caching via `pandas.HDFStore`:
    completed batches are written to disk and skipped on resume, so interrupted
    runs continue from the last completed batch.  Returns a concatenated
    DataFrame with a `'batch'` column.
  - `LFC_batch(...)`: deprecated alias for `gcate_lfc_batch`; emits
    `DeprecationWarning` and will be removed in a future release.
  - `subsample_ctrl_cells(ctrl_idx, n_ctrl=2000, random_state=0)` and
    `subsample_pert_cells(pert_idx, max_cells=2000, random_state=0)`:
    Internal utilities for reproducible, per-batch cell subsampling.
    `max_cells` caps pert cells only; ctrl cells are added on top.
  - Both batch functions accept DataFrame `A` with named perturbation columns.
  - `n_batches` parameter for `fit_gcate_batch` and `gcate_lfc_batch`:
    specifies total number of batches instead of per-batch count; overrides
    `batch_size` when set. Batches are sized evenly with `numpy.array_split`
    so the last batch is never more than 1 perturbation smaller than the
    others, avoiding an unstable single-pert tail batch.
  - `estimate_r(max_cells=N, random_state=0)`: new parameter that
    automatically subsamples to at most `N` cells before running JIC
    selection, prioritising control cells (all-zero rows of `A`). Filling
    the ctrl budget first and then drawing remaining slots from treated cells
    is equivalent to full-data estimation because confounding structure is
    concentrated in the baseline transcriptome.

### Fixed

- **Numba TBB fork warning**: Set `NUMBA_THREADING_LAYER_PRIORITY` to prefer OpenMP over TBB in `__init__.py`, eliminating `Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process` warnings when Joblib forks after Numba parallel execution.
- Added `llvm-openmp` to conda dependencies for fork-safe OpenMP threading.
- **Fast-path threshold** (`gcate_glm.py`): Raised the effective design-dimension ceiling and added a throughput heuristic; GCATE runs with larger `r` or wide batch designs now correctly use the crispyx path.
- **Backend toggle** (`gcate_glm.py`): Re-added `_USE_FAST_BACKEND` module flag and `_backend_override()` context manager; setting `_USE_FAST_BACKEND = False` now reliably forces statsmodels through the entire GCATE + LFC call graph.
- **Weighted dispersion** (`nb_glm_fast.py`): Dispersion averaging is now cell-count-weighted; low-coverage perturbations contribute proportionally less to the pooled estimate.
- **Control-cell residuals** (`nb_glm_fast.py`): Control-cell deviance residuals and fitted values are now initialised from the global covariate model, eliminating a last-perturbation overwrite bug.
- **Module-qualified imports** (`gcate_opt.py`, `gcate.py`, `DR_estimation.py`): Replaced import-time name bindings with module-qualified references so backend toggles propagate correctly at call time.
- **`estimate_r` bare name** (`gcate.py`): Fixed a `NameError` when calling `estimate_r` that was caused by a bare `fit_glm_auto` reference.
- **crispyx availability check** (`gcate_glm.py`): Added a one-time import check; users without crispyx now get a transparent fallback to statsmodels instead of a traceback.
- **Dead code removal** (`nb_glm_fast.py`): Removed an unreachable branch in `_compute_poisson_deviance_residuals`.

### Added

- **Fast GLM backend via crispyx** (`nb_glm_fast.py`)
  - `fit_glm_fast()`: Batch NB-GLM fitting using crispyx's `NBGLMBatchFitter`, replacing per-gene statsmodels IRLS with vectorized batch IRLS across all genes simultaneously.
  - `estimate_disp_fast()`: Vectorized method-of-moments dispersion estimation, replacing recursive Poisson GLM fitting.
  - `fit_glm_ondisk()`: On-disk streaming GLM fitting for large h5ad files.

- **Per-perturbation fitting strategy** (`_fit_glm_fast_per_perturbation()` in `nb_glm_fast.py`)
  - For multi-treatment data, loops over perturbation columns and fits binary (control vs. treatment_k) models, avoiding the memory and runtime cost of wide joint design matrices.
  - Covariate coefficients are averaged across perturbation-specific models; treatment coefficients are assembled into the full coefficient matrix.
  - Imputation is performed per-perturbation with correct counterfactual construction.

- **Auto-dispatch wrappers** (`gcate_glm.py`)
  - `fit_glm_auto()`: Routes to `fit_glm_fast()` when the effective design dimension is small and crispyx is available; falls back to statsmodels otherwise. Includes convergence checking with automatic fallback.
  - `estimate_disp_auto()`: Routes to `estimate_disp_fast()` when the number of genes is large enough for vectorized estimation to be faster.

### Changed

- **⚠️ `alter_min()` early-stopping defaults** (`gcate_opt.py`):
  - Default `kwargs_es['max_iters']` reduced from 500 → 50.
  - Default `tolerance` reduced from `1e-3` → `0.0` and a new scale-
    invariant `rel_tol=2e-4` introduced.  In practice the relative
    threshold plus `patience=5` reaches the same fixed point as the
    old absolute threshold within ≤ 50 iterations on every tested
    dataset, but external benchmarks that relied on the 500-iteration
    upper bound should pass `kwargs_es_1=dict(max_iters=500)` and
    `kwargs_es_2=dict(max_iters=500)` to recover the pre-v0.0.6 budget.

- **⚠️ BREAKING — `LFC()` variance and default `usevar`** (`DR_learner.py`):
  - Default `usevar` changed from `'pooled'` to `'unequal'` (Welch). Callers
    not specifying `usevar=` will see Welch t-statistics instead of pooled
    ones; revert with `LFC(..., usevar='pooled')` if reproducing pre-v0.0.6
    results is required.
  - **`'unequal'` formula corrected**: the variance is now
    `var = s₀²/n₀ + s₁²/n₁` (standard Welch); previously it was
    `(s₀²/n₀ + s₁²/n₁)/2`, a "half-Welch" that under-estimated the
    standard error by √2.  Test statistics are therefore ~√2× smaller
    than pre-v0.0.6 for explicit `usevar='unequal'` callers.
  - p-values now use the t-distribution with Welch-Satterthwaite degrees
    of freedom (per gene); the prior version used a Normal approximation.

- **`gcate_opt.py`**: `alter_min()` initialisation now uses the auto-dispatch GLM path.
- **`gcate.py`**: `_check_input()` and `estimate_r()` now use the auto-dispatch dispersion and GLM paths.
- **`DR_estimation.py`**: `cross_fitting()` now uses the auto-dispatch GLM path, enabling the per-perturbation fast path for multi-treatment LFC estimation.
- **`DR_learner.py`**: `LFC()` now accepts `backend: str = "auto"` as an explicit parameter; `"fast"` forces crispyx, `"original"` forces statsmodels.
- **`utils.py`**: `comp_size_factor()` vectorized with `np.nanmean`/`np.nanmedian`.

### Performance

Benchmarked on Perturb-seq data (n = 2,926 cells, p = 3,221 genes, 29 perturbations):

| Component | Original | Fast | Speedup |
|-----------|----------|------|---------|
| GCATE     | 331.6 s  | 298.5 s | 1.1×  |
| LFC       | 87.8 s   | 65.7 s  | 1.3×  |
| **Total** | **419.3 s** | **364.2 s** | **1.2×** |

On synthetic data (n = 500, p = 200): 61.5× GLM fit speedup, 7.1× imputation speedup.

Latent factor recovery: mean canonical correlation 0.998. LFC correlation: 0.856 (expected difference due to per-perturbation vs. joint model specification).

## [0.0.5] - 2025-01-30

- GCATE model for gene-level causal effect estimation from CRISPR screens.
- Doubly-robust learner (LFC, VIM) with AIPW pseudo-outcomes.
- Alternating minimization with Numba+OpenMP acceleration.
- Statsmodels-based per-gene NB-GLM fitting.
- Multiple testing correction (BH, step-down, FDX).
