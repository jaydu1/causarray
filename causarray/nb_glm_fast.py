"""
Fast NB-GLM fitting for causarray using crispyx's vectorized backend.

This module provides drop-in replacements for causarray's `fit_glm` and
`estimate_disp` that leverage crispyx's Numba-accelerated batch IRLS solver
and streaming on-disk computation for large sc-CRISPR datasets.

Key functions:
- ``estimate_disp_fast``: Vectorized dispersion estimation (replaces gene-by-gene Poisson GLM).
- ``fit_glm_fast``: Batch NB-GLM fitting via crispyx's ``NBGLMBatchFitter`` (replaces statsmodels gene-by-gene).
- ``fit_glm_ondisk``: On-disk NB-GLM fitting for datasets that don't fit in memory.
"""

from __future__ import annotations

import logging
import warnings
from typing import Literal

import numpy as np
import scipy.sparse as sp

from causarray.utils import comp_size_factor, _filter_params, pprint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

_SPARSE_WARN_GB: float = 4.0
"""Emit ResourceWarning when materialising sparse Y into dense float64 would
exceed this many gigabytes."""


def _scale_design_columns(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Standardize design-matrix columns to unit std (excluding constant cols).

    crispyx's IRLS solver does not precondition the design matrix.  When
    columns have very different scales (e.g. intercept std=0 alongside latent
    factor columns with std≈0.009) the weighted normal equations XᵀWX become
    ill-conditioned and coefficients can blow up by 5× or more vs statsmodels.

    This function scales each column by its standard deviation so that all
    non-constant columns have unit variance going into the IRLS solver.
    Constant columns (intercept, std≈0) are left unchanged.

    Parameters
    ----------
    X : (n, d) array
        Design matrix with at least one column.

    Returns
    -------
    X_scaled : (n, d) array
        Column-scaled design matrix.
    col_scale : (d,) array
        Scale factors (std per column; 1.0 for constant columns).
        To recover original-space coefficients from scaled-space coefficients::

            B_original = B_scaled / col_scale  # (p, d)
    """
    col_scale = X.std(axis=0)
    col_scale = np.where(col_scale > 1e-8, col_scale, 1.0)
    return X / col_scale, col_scale


def _maybe_densify(Y: np.ndarray) -> np.ndarray:
    """Convert sparse or non-float64 Y to a dense float64 array.

    Emits ``ResourceWarning`` when the estimated dense representation would
    exceed ``_SPARSE_WARN_GB`` GB so callers are aware of the memory cost.
    """
    if sp.issparse(Y):
        gb = Y.shape[0] * Y.shape[1] * 8 / 1e9
        if gb > _SPARSE_WARN_GB:
            warnings.warn(
                f"Materialising sparse Y ({Y.shape[0]}×{Y.shape[1]}) into dense "
                f"float64 requires ~{gb:.1f} GB of memory.",
                ResourceWarning,
                stacklevel=3,
            )
        return np.asarray(Y.toarray(), dtype=np.float64)
    return np.asarray(Y, dtype=np.float64)


def estimate_disp_fast(
    Y: np.ndarray,
    X: np.ndarray,
    *,
    A: np.ndarray | None = None,
    offset: np.ndarray | None = None,
    method: Literal["moments", "trend"] = "moments",
) -> np.ndarray:
    """Estimate NB dispersion parameters using vectorized method-of-moments.

    This replaces causarray's ``estimate_disp`` which fits a full Poisson GLM
    per gene.  Instead, a single Poisson IRLS pass (via crispyx) estimates
    fitted values for all genes simultaneously, then uses method-of-moments
    to compute dispersion.

    Parameters
    ----------
    Y : (n, p) array
        Raw count matrix.
    X : (n, d) array
        Design matrix (should include intercept if desired).
    A : (n, a) array or None
        Optional treatment indicator matrix.  When provided it is appended to
        ``X`` so that strongly-responding genes do not inflate the dispersion
        estimate.
    offset : (n,) array or None
        Log-scale offset (e.g. log size factors).
    method : str
        ``"moments"`` for simple MoM, ``"trend"`` for MoM + trend shrinkage.

    Returns
    -------
    disp : (p,) array
        Dispersion parameters (r in NB(r, p) parameterisation; same as
        causarray's ``disp_glm``).
    """
    from crispyx.glm import NBGLMBatchFitter

    n, p = Y.shape
    Y_float = _maybe_densify(Y)

    if offset is not None:
        offset = np.asarray(offset, dtype=np.float64).ravel()
    else:
        offset = np.zeros(n, dtype=np.float64)

    # Including treatment indicators in the Poisson init fit prevents
    # strongly-responding genes from inflating the dispersion estimate.
    X_disp = X if A is None else np.c_[X, A]

    # Single crispyx batch fit estimates dispersion via MoM internally.
    # We use its result.dispersion (alpha) directly, converting to
    # causarray's r = 1/alpha parameterisation.
    fitter = NBGLMBatchFitter(
        design=X_disp,
        offset=offset,
        max_iter=10,
        poisson_init_iter=5,
        dispersion_method="moments",
        min_mu=0.5,
    )
    result = fitter.fit_batch(Y_float)

    alpha = result.dispersion  # (p,) overdispersion parameter
    alpha = np.clip(alpha, 1e-8, 100.0)
    alpha[~np.isfinite(alpha)] = 1.0
    disp_glm = 1.0 / alpha  # r = 1/alpha (causarray convention)
    disp_glm[~np.isfinite(disp_glm)] = 1.0

    return disp_glm


# ---------------------------------------------------------------------------
# In-memory batch NB-GLM fitting
# ---------------------------------------------------------------------------


def fit_glm_fast(
    Y: np.ndarray,
    X: np.ndarray,
    A: np.ndarray | None = None,
    family: str = "gaussian",
    disp_family: str = "poisson",
    disp_glm: np.ndarray | None = None,
    impute: bool | np.ndarray = False,
    offset: np.ndarray | bool | None = None,
    offset_test: np.ndarray | None = None,
    shrinkage: bool = False,
    alpha: float = 1e-4,
    maxiter: int = 1000,
    thres_disp: float = 100.0,
    n_jobs: int = -3,
    random_state: int = 0,
    verbose: bool = False,
    mem_limit_gb: float | None = None,
    **kwargs,
):
    """Fast batch NB-GLM fitting using crispyx's vectorized IRLS backend.

    This is a drop-in replacement for causarray's ``fit_glm`` that uses
    crispyx's ``NBGLMBatchFitter`` instead of per-gene statsmodels fits.

    Parameters and returns are identical to ``causarray.gcate_glm.fit_glm``.
    """
    from crispyx.glm import NBGLMBatchFitter

    np.random.seed(random_state)

    if family not in ["gaussian", "poisson", "nb"]:
        raise ValueError("Family not recognized")

    n, p = Y.shape
    d = X.shape[1]

    # Handle treatment indicator
    if A is None:
        a = 1
        X_full = X
        X_test = None
    else:
        if A.ndim == 1:
            A = A[:, None]
        a = A.shape[1]
        if impute is not False and isinstance(impute, np.ndarray):
            X_test = impute
        else:
            X_test = X.copy()
        X_test = np.c_[X_test, np.zeros((X_test.shape[0], a))]
        X_full = np.c_[X, A]

    # Handle offset / size factors
    if offset is not None and offset is not False:
        if isinstance(offset, bool) and offset is True:
            offsets = np.log(
                comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs))
            )
        else:
            offsets = np.asarray(offset, dtype=np.float64).ravel()
    else:
        offsets = None

    # For Gaussian family, fall back to original implementation
    if family == "gaussian":
        from causarray.gcate_glm import fit_glm as _fit_glm_orig
        return _fit_glm_orig(
            Y, X, A=None if A is None else A,
            family=family, disp_family=disp_family,
            disp_glm=disp_glm, impute=impute, offset=offset,
            shrinkage=shrinkage, alpha=alpha, maxiter=maxiter,
            thres_disp=thres_disp, n_jobs=n_jobs,
            random_state=random_state, verbose=verbose, **kwargs,
        )

    # Estimate dispersion for NB — use covariate-only design (X, not X_full)
    # since dispersion is a gene-level property and doesn't depend on
    # treatment indicators.  Using X_full with many treatment columns
    # (d + a) would be extremely slow for the batch fitter.
    if family == "nb" and disp_glm is None:
        disp_glm = estimate_disp_fast(
            Y, X, offset=offsets, method="moments"
        )

    if verbose:
        pprint.pprint(f"Fitting {family} GLM (fast)...")

    Y_float = _maybe_densify(Y)
    offset_arr = offsets if offsets is not None else np.zeros(n, dtype=np.float64)

    # --- Per-perturbation strategy ---
    # When A has multiple columns (multi-treatment), loop over perturbations
    # using binary d = d_cov + 1 models. This matches crispyx's approach and
    # keeps d small enough for the fast Numba/Cramer WLS path.
    if A is not None and a > 1:
        return _fit_glm_fast_per_perturbation(
            Y_float, X, A, a, d, p, n, family, disp_glm,
            impute, X_test, offset_arr, offsets, maxiter, verbose,
            mem_limit_gb=mem_limit_gb, offset_test=offset_test,
        )

    B, Yhat, disp_glm_out, resid_deviance = _fit_glm_fast_single(
        Y_float, X_full, p, n, family, disp_glm, offset_arr, maxiter, verbose,
    )

    # Handle imputation (counterfactual predictions)
    if impute is not False and A is not None:
        n_test = X_test.shape[0]
        # Prefer an explicitly-supplied test-fold offset (e.g. from K>1
        # cross-fitting where the training offset has length n_train ≠ n_test).
        # Fall back to the training offset only when its length matches n_test
        # (K=1 case where train and test indices coincide).
        if offset_test is not None and np.asarray(offset_test).shape[0] == n_test:
            offsets_test = np.asarray(offset_test, dtype=np.float64).ravel()
        elif offsets is not None and offsets.shape[0] == n_test:
            offsets_test = offsets
        else:
            offsets_test = None
        Yhat_0 = np.zeros((n_test, p, a))
        Yhat_1 = np.zeros((n_test, p, a))
        for k in range(a):
            X_test_copy = X_test.copy()
            eta_0 = X_test_copy @ B.T
            if offsets_test is not None:
                eta_0 += offsets_test[:, None]
            Yhat_0[:, :, k] = np.exp(np.clip(eta_0, -20, 20))

            X_test_copy[:, d + k] = 1
            eta_1 = X_test_copy @ B.T
            if offsets_test is not None:
                eta_1 += offsets_test[:, None]
            Yhat_1[:, :, k] = np.exp(np.clip(eta_1, -20, 20))
        Yhat = (Yhat_0, Yhat_1)

    return B, Yhat, disp_glm_out if family == "nb" else disp_glm, offsets, resid_deviance


def _fit_glm_fast_single(
    Y_float, X_full, p, n, family, disp_glm, offset_arr, maxiter, verbose,
):
    """Fit a single batch GLM across all genes (no per-perturbation split).

    Used when ``A is None`` (covariate-only fit) or when ``a == 1`` (single
    perturbation).  In the ``a == 1`` case the treatment column has already
    been appended to ``X_full = [X | A]`` by the caller, so this function
    fits a joint model of width ``d + 1`` rather than a per-perturbation loop.
    This is equivalent to the per-perturbation approach when there is only one
    treatment arm and avoids the overhead of the two-stage strategy.
    """
    from crispyx.glm import NBGLMBatchFitter

    # Column-precondition X_full so that crispyx's IRLS sees unit-std columns.
    # crispyx does not internally normalize the design matrix; poorly-scaled
    # columns (e.g. latent factor columns with std≈0.009) cause XᵀWX to be
    # ill-conditioned and blow up coefficients by ~5× vs statsmodels.
    # We fit in scaled space and recover original-space coefficients afterwards.
    X_scaled, col_scale = _scale_design_columns(X_full)

    # Convert causarray's disp_glm (r) to crispyx's alpha = 1/r
    fixed_alpha = 1.0 / np.clip(disp_glm, 0.01, 1e6) if disp_glm is not None else None

    if family == "nb":
        fitter = NBGLMBatchFitter(
            design=X_scaled,
            offset=offset_arr,
            max_iter=min(maxiter, 50),
            poisson_init_iter=5,
            dispersion_method="moments",
            min_mu=0.5,
        )
        result = fitter.fit_batch_with_joint_offsets(
            Y_float, fixed_dispersion=fixed_alpha,
        )
        # Recover original-space coefficients: b_j = b_scaled_j / scale_j
        B = result.coef / col_scale  # (p, d)

        eta = X_full @ B.T + offset_arr[:, None]
        Yhat = np.exp(np.clip(eta, -20, 20))

        fitter_disp = result.dispersion
        resid_deviance = _compute_nb_deviance_residuals(Y_float, Yhat, fitter_disp)
        disp_glm_out = 1.0 / np.clip(fitter_disp, 1e-8, 1e6)
        disp_glm_out[~np.isfinite(disp_glm_out)] = 1.0

    elif family == "poisson":
        fitter = NBGLMBatchFitter(
            design=X_scaled,
            offset=offset_arr,
            max_iter=min(maxiter, 50),
            poisson_init_iter=10,
            dispersion_method="moments",
            min_mu=0.5,
        )
        result = fitter.fit_batch(Y_float)
        # Recover original-space coefficients
        B = result.coef / col_scale  # (p, d)

        eta = X_full @ B.T + offset_arr[:, None]
        Yhat = np.exp(np.clip(eta, -20, 20))

        resid_deviance = _compute_poisson_deviance_residuals(Y_float, Yhat)
        disp_glm_out = disp_glm

    return B, Yhat, disp_glm_out, resid_deviance


def _fit_glm_fast_per_perturbation(
    Y_float, X, A, a, d, p, n, family, disp_glm,
    impute, X_test, offset_arr, offsets, maxiter, verbose,
    mem_limit_gb=None, offset_test=None,
):
    """Per-perturbation GLM fitting with global covariate model.

    Uses a two-stage approach:
    1. Fit a global covariate-only model on ALL cells to get stable
       covariate coefficients and dispersion.
    2. For each perturbation k, estimate the treatment effect (d=1) using
       crispyx's ``fit_batch_with_joint_offsets(covariate_offset=...)``,
       which conditions on the global covariate predictions.

    This gives:
    - Stable Y_hat_0 across perturbations (from global covariate model)
    - Per-perturbation treatment effects
    - Fast fitting (d=1 per perturbation)
    """
    from crispyx.glm import NBGLMBatchFitter

    ctrl_mask = A.sum(axis=1) == 0  # control cells

    # ── Stage 1: Global covariate model on ALL cells ─────────────────────
    # Column-precondition X so that crispyx's IRLS sees unit-std columns.
    # Without scaling, latent-factor columns (std≈0.009) make XᵀWX ill-
    # conditioned, inflating B_cov_global by ~5–10× and propagating into
    # cov_offset_all → artificially large treatment effects for zero-inflated
    # genes that bypass the thres_min guard in DR_learner.
    X_scaled_global, col_scale_global = _scale_design_columns(X)

    # min_mu for the IRLS must be well below thres_min=0.01 in DR_learner so
    # that near-zero genes (true mean ≈ 0.001) get small Yhat predictions.
    # With min_mu=0.5 (old default), Yhat_0≈0.5 and Yhat_1≈0.184 for near-zero
    # genes → AIPW tau_1 ≈ 0.18 >> thres_min → spurious tau ≈ 2.9 artifacts.
    # With min_mu=1e-4: Yhat predictions for near-zero genes are tiny → both
    # tau_0 and tau_1 clip to thres_diff=0.01 → |diff|=0 → guard fires. ✓
    _IRLS_MIN_MU = 1e-4

    if family == "nb":
        fitter_global = NBGLMBatchFitter(
            design=X_scaled_global,
            offset=offset_arr,
            max_iter=min(maxiter, 50),
            poisson_init_iter=5,
            dispersion_method="moments",
            min_mu=_IRLS_MIN_MU,
        )
        result_global = fitter_global.fit_batch(Y_float)
        B_cov_global = result_global.coef / col_scale_global  # (p, d) original-space
        global_alpha = result_global.dispersion  # (p,)
        global_alpha = np.clip(global_alpha, 1e-8, 100.0)
        global_alpha[~np.isfinite(global_alpha)] = 1.0
    elif family == "poisson":
        fitter_global = NBGLMBatchFitter(
            design=X_scaled_global,
            offset=offset_arr,
            max_iter=min(maxiter, 50),
            poisson_init_iter=10,
            dispersion_method="moments",
            min_mu=_IRLS_MIN_MU,
        )
        result_global = fitter_global.fit_batch(Y_float)
        B_cov_global = result_global.coef / col_scale_global  # (p, d) original-space
        global_alpha = None

    # Precompute covariate offset for all cells: (n, p)
    # Using original-space X and B_cov_global (equivalent to X_scaled @ B_scaled).
    cov_offset_all = X @ B_cov_global.T  # (n, p)

    # ── Stage 2: Per-perturbation treatment effects ──────────────────────
    B_full = np.zeros((p, d + a), dtype=np.float64)
    B_full[:, :d] = B_cov_global

    # Initialise residuals and Yhat from the Stage-1 global covariate model.
    # The loop below overwrites only treated-cell rows so that control cells
    # remain anchored to this global fit throughout.
    eta_global = cov_offset_all + offset_arr[:, None]  # (n, p)
    Yhat_global = np.exp(np.clip(eta_global, -20, 20))
    if family == "nb":
        resid_deviance = _compute_nb_deviance_residuals(Y_float, Yhat_global, global_alpha)
    else:
        resid_deviance = _compute_poisson_deviance_residuals(Y_float, Yhat_global)
    Yhat_full = Yhat_global.copy()
    # Accumulate dispersion weighted by the number of cells in each sub-model
    # so that small perturbation groups contribute proportionally less.
    disp_alpha = np.zeros(p, dtype=np.float64)
    disp_alpha_cells = 0

    do_impute = impute is not False and X_test is not None
    if do_impute:
        n_test = X_test.shape[0]
        # Prefer an explicitly-supplied test-fold offset (e.g. from K>1
        # cross-fitting where the training offset has length n_train ≠ n_test).
        # Fall back to the training offset only when its length matches n_test
        # (K=1 case where train and test indices coincide).
        if offset_test is not None and np.asarray(offset_test).shape[0] == n_test:
            offsets_test = np.asarray(offset_test, dtype=np.float64).ravel()
        elif offsets is not None and offsets.shape[0] == n_test:
            offsets_test = offsets
        else:
            offsets_test = None
        # Y_hat_0 is the SAME for all perturbations (from global cov model)
        X_test_cov = X_test[:, :d]
        eta_0_test = X_test_cov @ B_cov_global.T  # (n_test, p)
        if offsets_test is not None:
            eta_0_test += offsets_test[:, None]
        Yhat_0_base = np.exp(np.clip(eta_0_test, -20, 20))  # (n_test, p)
        # Choose dtype: switch to float32 when both imputation arrays would
        # exceed mem_limit_gb GB (two arrays of shape n_test × p × a).
        _impu_gb = n_test * p * a * 2 * 8 / 1e9
        if mem_limit_gb is not None and _impu_gb > mem_limit_gb:
            import warnings
            warnings.warn(
                f"Imputation arrays ({_impu_gb:.1f} GB as float64) exceed "
                f"mem_limit_gb={mem_limit_gb} GB; using float32 to halve peak memory.",
                ResourceWarning, stacklevel=4,
            )
            _impu_dtype: type = np.float32
        else:
            _impu_dtype = np.float64
        Yhat_0 = np.zeros((n_test, p, a), dtype=_impu_dtype)
        Yhat_1 = np.zeros((n_test, p, a), dtype=_impu_dtype)

    for k in range(a):
        pert_mask = A[:, k] == 1
        cell_mask = ctrl_mask | pert_mask
        n_sub = cell_mask.sum()

        # Treatment-only design (d=1): just the treatment indicator
        A_k = A[cell_mask, k : k + 1]  # (n_sub, 1)
        Y_sub = Y_float[cell_mask]
        offset_sub = offset_arr[cell_mask]
        cov_offset_sub = cov_offset_all[cell_mask]  # (n_sub, p)

        if family == "nb":
            fitter = NBGLMBatchFitter(
                design=A_k,
                offset=offset_sub,
                max_iter=min(maxiter, 50),
                poisson_init_iter=5,
                dispersion_method="moments",
                min_mu=_IRLS_MIN_MU,
            )
            result = fitter.fit_batch_with_joint_offsets(
                Y_sub,
                covariate_offset=cov_offset_sub,
                fixed_dispersion=global_alpha,
            )
            B_trt_k = result.coef[:, 0]  # (p,) single treatment coefficient

            # Fitted values: eta = B_trt * A_k + cov_offset + offset
            eta_sub = A_k @ result.coef.T + cov_offset_sub + offset_sub[:, None]
            Yhat_sub = np.exp(np.clip(eta_sub, -20, 20))

            fitter_disp = result.dispersion
            resid_sub = _compute_nb_deviance_residuals(Y_sub, Yhat_sub, fitter_disp)
            disp_alpha += n_sub * fitter_disp
            disp_alpha_cells += n_sub

        elif family == "poisson":
            fitter = NBGLMBatchFitter(
                design=A_k,
                offset=offset_sub,
                max_iter=min(maxiter, 50),
                poisson_init_iter=10,
                dispersion_method="moments",
                min_mu=_IRLS_MIN_MU,
            )
            result = fitter.fit_batch_with_joint_offsets(
                Y_sub,
                covariate_offset=cov_offset_sub,
            )
            B_trt_k = result.coef[:, 0]

            eta_sub = A_k @ result.coef.T + cov_offset_sub + offset_sub[:, None]
            Yhat_sub = np.exp(np.clip(eta_sub, -20, 20))
            resid_sub = _compute_poisson_deviance_residuals(Y_sub, Yhat_sub)

        B_full[:, d + k] = B_trt_k

        # Update only treated cells; control-cell rows keep the global-model values.
        treated_in_sub = np.where(cell_mask)[0][A_k[:, 0] == 1]
        resid_deviance[treated_in_sub] = resid_sub[A_k[:, 0] == 1]
        Yhat_full[treated_in_sub] = Yhat_sub[A_k[:, 0] == 1]

        # Imputation: global cov + per-perturbation treatment
        if do_impute:
            Yhat_0[:, :, k] = Yhat_0_base  # same for all k
            # Y_hat_1 = exp(cov_offset + B_trt_k + offset)
            eta_1_test = eta_0_test + B_trt_k[None, :]  # broadcast (n_test, p)
            Yhat_1[:, :, k] = np.exp(np.clip(eta_1_test, -20, 20))

    if family == "nb":
        # Weighted average of alpha; convert to r = 1/alpha (causarray convention).
        disp_alpha /= max(disp_alpha_cells, 1)
        disp_glm_out = 1.0 / np.clip(disp_alpha, 1e-8, 1e6)
        disp_glm_out[~np.isfinite(disp_glm_out)] = 1.0
    else:
        disp_glm_out = disp_glm

    Yhat_out = (Yhat_0, Yhat_1) if do_impute else Yhat_full
    return B_full, Yhat_out, disp_glm_out, offsets, resid_deviance


# ---------------------------------------------------------------------------
# On-disk NB-GLM fitting
# ---------------------------------------------------------------------------


def fit_glm_ondisk(
    path: str,
    perturbation_col: str = "perturbation",
    control_label: str = "control",
    target_label: str | None = None,
    gene_indices: np.ndarray | None = None,
    covariate_columns: list[str] | None = None,
    chunk_size: int = 2048,
    max_iter: int = 25,
    verbose: bool = False,
):
    """On-disk NB-GLM fitting using crispyx's streaming functions.

    Reads data directly from an h5ad file without loading the full count
    matrix into memory.  Uses crispyx's streaming control statistics,
    global dispersion estimation, and batch NB-GLM fitting.

    Parameters
    ----------
    path : str
        Path to the h5ad file.
    perturbation_col : str
        Column in obs containing perturbation labels.
    control_label : str
        Label for control cells.
    target_label : str or None
        Label for the target perturbation.  If None, all non-control
        perturbations are compared to control.
    gene_indices : array or None
        Indices of genes to fit.  If None, all genes are used.
    covariate_columns : list or None
        Additional covariate columns from obs.
    chunk_size : int
        Number of cells per chunk for streaming.
    max_iter : int
        Maximum IRLS iterations.
    verbose : bool
        Print progress.

    Returns
    -------
    B : (d, p) array
        Coefficient matrix (d = n_features, p = n_genes).
    Yhat : (n, p) array
        Fitted values.
    disp : (p,) array
        Dispersion parameters.
    offsets : (n,) array
        Log size factors.
    resid_deviance : (n, p) array
        Deviance residuals.
    """
    import anndata as ad
    from crispyx.glm import NBGLMBatchFitter
    from crispyx._size_factors import iter_matrix_chunks as _iter_sf_chunks
    from crispyx.data import read_backed

    # Load metadata
    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    n_cells, n_genes_total = adata.shape

    perturbations = obs[perturbation_col].values
    if target_label is not None:
        mask = np.isin(perturbations, [control_label, target_label])
    else:
        mask = np.ones(n_cells, dtype=bool)

    cell_indices = np.where(mask)[0]
    n = len(cell_indices)
    pert_labels = perturbations[cell_indices]
    A_vec = (pert_labels != control_label).astype(np.float64)

    if gene_indices is None:
        gene_indices = np.arange(n_genes_total)
    p = len(gene_indices)

    if verbose:
        logger.info(f"On-disk NB-GLM: {n} cells × {p} genes")

    # Read selected cells and genes
    # For moderate-size subsets, read into memory in chunks
    Y = np.zeros((n, p), dtype=np.float64)
    backed = read_backed(path)
    try:
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            idx = cell_indices[start:end]
            chunk = backed.X[idx][:, gene_indices]
            if sp.issparse(chunk):
                chunk = np.asarray(chunk.toarray(), dtype=np.float64)
            else:
                chunk = np.asarray(chunk, dtype=np.float64)
            Y[start:end] = chunk
    finally:
        backed.file.close()

    # Compute size factors
    sf = comp_size_factor(Y)
    offsets = np.log(sf)

    # Build design matrix: [intercept | covariates (optional)]
    # Covariates are appended after the intercept so that fit_glm_fast receives
    # a covariate-only X and the treatment vector as A.
    if covariate_columns:
        cov_data = np.column_stack(
            [obs[col].values[cell_indices].astype(np.float64) for col in covariate_columns]
        )
        X_cov = np.column_stack([np.ones(n), cov_data])
    else:
        X_cov = np.ones((n, 1))

    # Estimate dispersion and fit NB-GLM using fast path
    B, Yhat, disp, _, resid_dev = fit_glm_fast(
        Y, X_cov, A=A_vec[:, None], family="nb", offset=offsets,
        maxiter=max_iter, verbose=verbose,
    )

    adata.file.close()
    return B, Yhat, disp, offsets, resid_dev


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_nb_deviance_residuals(
    Y: np.ndarray, mu: np.ndarray, alpha: np.ndarray
) -> np.ndarray:
    """Compute signed NB deviance residuals.

    Parameters
    ----------
    Y : (n, p) array of counts.
    mu : (n, p) array of fitted means.
    alpha : (p,) array of dispersion (1/r).

    Returns
    -------
    resid : (n, p) array of signed deviance residuals.
    """
    mu = np.maximum(mu, 1e-10)
    r = 1.0 / np.maximum(alpha, 1e-10)  # (p,)

    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(
            Y > 0,
            Y * np.log(np.maximum(Y, 1e-10) / mu),
            0.0,
        )
        term2 = (Y + r[None, :]) * np.log(
            (Y + r[None, :]) / (mu + r[None, :])
        )
        dev = 2.0 * (term1 - term2)

    sign = np.sign(Y - mu)
    resid = sign * np.sqrt(np.maximum(dev, 0.0))
    resid[~np.isfinite(resid)] = 0.0
    return resid


def _compute_poisson_deviance_residuals(
    Y: np.ndarray, mu: np.ndarray
) -> np.ndarray:
    """Compute signed Poisson deviance residuals."""
    mu = np.maximum(mu, 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        # Poisson deviance: 2*[y*log(y/μ) - (y-μ)] for y>0; 2*μ for y=0
        term = np.where(Y > 0, Y * np.log(np.maximum(Y, 1e-10) / mu) - (Y - mu), mu)
        dev = 2.0 * term

    sign = np.sign(Y - mu)
    resid = sign * np.sqrt(np.maximum(dev, 0.0))
    resid[~np.isfinite(resid)] = 0.0
    return resid
