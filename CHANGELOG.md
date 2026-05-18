# Changelog

## [0.0.6]

### Fixed

- **Numba TBB fork warning**: Set `NUMBA_THREADING_LAYER_PRIORITY` to prefer OpenMP over TBB in `__init__.py`, eliminating `Numba: Attempted to fork from a non-main thread, the TBB library may be in an invalid state in the child process` warnings when Joblib forks after Numba parallel execution.
- Added `llvm-openmp` to conda dependencies for fork-safe OpenMP threading.
- **Fast-path threshold** (`gcate_glm.py`): Raised `d_eff` ceiling from 15 → 30 (`_FAST_MAX_D`) and added throughput heuristic `n×p/d² > 5 000`; GCATE runs with `r ≥ 10` or wide batch designs now correctly use the crispyx path.
- **Backend toggle** (`gcate_glm.py`): Re-added `_USE_FAST_BACKEND` module flag and `_backend_override()` context manager; setting `_USE_FAST_BACKEND = False` now reliably forces statsmodels through the entire GCATE + LFC call graph.
- **Weighted dispersion** (`nb_glm_fast.py`): Dispersion averaging in `_fit_glm_fast_per_perturbation` is now cell-count-weighted instead of unweighted; low-coverage perturbations contribute proportionally less to the pooled dispersion estimate.
- **Control-cell residuals** (`nb_glm_fast.py`): Control-cell deviance residuals and fitted values in `_fit_glm_fast_per_perturbation` are now initialised from the Stage-1 global covariate model; the loop overwrites only treated-cell rows, eliminating the last-perturbation overwrite bug.
- **Module-qualified imports** (`gcate_opt.py`, `gcate.py`, `DR_estimation.py`): Replaced import-time name bindings of `fit_glm_auto` / `estimate_disp_auto` with module-qualified references (`_gcate_glm.fit_glm_auto`) so backend toggles propagate correctly at call time.
- **crispyx availability check** (`gcate_glm.py`): Added `_CRISPYX_AVAILABLE` flag (evaluated once at import time) and `ImportError` guard inside the fast path; users without crispyx now get a transparent fallback to statsmodels instead of a traceback.
- **Dead code removal** (`nb_glm_fast.py`): Removed unreachable first `np.where` assignment in `_compute_poisson_deviance_residuals` (Y=0 branch evaluated to `-mu + mu = 0` and was silently overwritten).

### Added

- **Fast GLM backend via crispyx** (`nb_glm_fast.py`)
  - `fit_glm_fast()`: Batch NB-GLM fitting using crispyx's `NBGLMBatchFitter`, replacing per-gene statsmodels IRLS with vectorized batch IRLS across all genes simultaneously.
  - `estimate_disp_fast()`: Vectorized method-of-moments dispersion estimation, replacing recursive Poisson GLM fitting.
  - `fit_glm_ondisk()`: On-disk streaming GLM fitting for large h5ad files.

- **Per-perturbation fitting strategy** (`_fit_glm_fast_per_perturbation()` in `nb_glm_fast.py`)
  - For multi-treatment data, loops over perturbation columns and fits binary (control vs. treatment_k) models with design dimension d = d_cov + 1.
  - Avoids crispyx's O(n × p × d²) einsum WLS bottleneck that occurs with wide joint design matrices (d = 30–40).
  - Covariate coefficients are averaged across perturbation-specific models; treatment coefficients are assembled into the full coefficient matrix.
  - Imputation is performed per-perturbation with correct counterfactual construction.

- **Auto-dispatch wrappers** (`gcate_glm.py`)
  - `fit_glm_auto()`: Routes to `fit_glm_fast()` when effective design dimension d_eff ≤ 15 and crispyx is available; falls back to statsmodels `fit_glm()` otherwise. Includes convergence checking with automatic fallback.
  - `estimate_disp_auto()`: Routes to `estimate_disp_fast()` when p ≥ 50.

### Changed

- **`gcate_opt.py`**: `alter_min()` initialization now calls `fit_glm_auto()` instead of `fit_glm()`.
- **`gcate.py`**: `_check_input()` now calls `estimate_disp_auto()` instead of `estimate_disp()`; `estimate_r()` now calls `fit_glm_auto()`.
- **`DR_estimation.py`**: `cross_fitting()` now calls `fit_glm_auto()`, enabling the per-perturbation fast path for multi-treatment LFC estimation.
- **`utils.py`**: `comp_size_factor()` vectorized with `np.nanmean`/`np.nanmedian`, replacing slow `np.apply_along_axis` loop.

### Performance

Benchmarked on Perturb-seq data (n = 2,926 cells, p = 3,221 genes, 29 perturbations):

| Component | Original | Fast | Speedup |
|-----------|----------|------|---------|
| GCATE     | 331.6 s  | 298.5 s | 1.1×  |
| LFC       | 87.8 s   | 65.7 s  | 1.3×  |
| **Total** | **419.3 s** | **364.2 s** | **1.2×** |

On synthetic data (n = 500, p = 200): 61.5× GLM fit speedup, 7.1× imputation speedup.

Latent factor recovery: mean canonical correlation 0.998. LFC correlation: 0.856 (expected difference due to per-perturbation vs. joint model specification).

## [0.0.5] - Initial Release

- GCATE model for gene-level causal effect estimation from CRISPR screens.
- Doubly-robust learner (LFC, VIM) with AIPW pseudo-outcomes.
- Alternating minimization with Numba+OpenMP acceleration.
- Statsmodels-based per-gene NB-GLM fitting.
- Multiple testing correction (BH, step-down, FDX).
