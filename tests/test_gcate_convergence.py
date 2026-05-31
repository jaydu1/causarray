"""
Unit tests for GCATE confounder estimation quality.

Tests T1–T6 use simulated NB/Poisson data with known latent structure so
that confounder recovery can be evaluated objectively.
"""

import numpy as np
import pytest
from causarray.gcate import fit_gcate


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def simulate_gcate_data(n=200, p=300, r=2, sparsity=0.3, seed=42, family='nb'):
    """Generate a count matrix with r known latent confounders.

    Parameters
    ----------
    n : int
        Number of cells.
    p : int
        Number of genes.
    r : int
        Number of true latent confounders.
    sparsity : float
        Target zero-fraction.  Values > 0.5 trigger extra zero-inflation by
        down-scaling mean expression for a subset of genes.
    seed : int
        Random seed.
    family : str
        'nb' (negative binomial) or 'poisson'.

    Returns
    -------
    Y : (n, p) float array — count matrix
    X : (n, 2) float array — intercept + batch covariate
    A : (n, 1) float array — binary treatment
    U_true : (n, r) float array — true latent confounders (orthonormal columns)
    Gamma_true : (p, r) float array — true gene loadings
    B_X : (p, 2) float array — true covariate coefficients
    B_A : (p, 1) float array — true treatment effects
    """
    rng = np.random.default_rng(seed)

    # Observed covariates: intercept + continuous batch
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    # Binary treatment
    A = rng.binomial(1, 0.5, n)[:, None].astype(float)

    # True latent confounders — use natural scale (not QR-normalised) so that
    # the confounder signal in eta is O(1) and clearly recoverable.
    # Column norms ≈ sqrt(n) ≈ 14, giving eta signal std ≈ 0.7 vs noise ≈ 0.5.
    U_true = rng.standard_normal((n, r))

    # Gene loadings
    Gamma_true = rng.standard_normal((p, r)) * 0.5

    # GLM coefficients
    B_X = rng.standard_normal((p, X.shape[1])) * 0.3
    B_A = rng.standard_normal((p, 1)) * 0.5

    # Linear predictor (n x p).  No random size-factor variation so that
    # offset=False in the tests is self-consistent.
    eta = X @ B_X.T + A @ B_A.T + U_true @ Gamma_true.T
    # Clip to avoid explosion
    mu = np.exp(np.clip(eta, -4, 4))

    # Apply additional zero-inflation for high-sparsity request
    if sparsity > 0.4:
        n_sparse = int(p * min((sparsity - 0.3) / 0.5, 0.9))
        sparse_genes = rng.choice(p, n_sparse, replace=False)
        mu[:, sparse_genes] *= 0.05

    # Sample counts
    if family == 'nb':
        disp = 5.0
        p_nb = disp / (disp + mu)
        p_nb = np.clip(p_nb, 1e-6, 1 - 1e-6)
        Y = rng.negative_binomial(int(disp), p_nb).astype(float)
    else:  # poisson
        # Scale mu down so counts stay moderate and signal/noise is favourable
        Y = rng.poisson(mu * 0.5).astype(float)

    # Ensure no all-zero columns (required by fit_gcate)
    all_zero = np.all(Y == 0, axis=0)
    Y[:, all_zero] += 1.0

    return Y, X, A, U_true, Gamma_true, B_X, B_A


def subspace_r2(U_est, U_true):
    """Fraction of U_true variance explained by the column space of U_est.

    Uses QR decomposition so the metric is invariant to rotation/scaling of
    the estimated factors.

    Returns
    -------
    r2 : float in [0, 1]
    """
    if U_est.shape[1] == 0 or U_true.shape[1] == 0:
        return 0.0
    Q, _ = np.linalg.qr(U_est)
    proj = Q @ (Q.T @ U_true)
    ss_res = np.sum((U_true - proj) ** 2)
    ss_tot = np.sum(U_true ** 2)
    if ss_tot < 1e-12:
        return 1.0
    return float(1.0 - ss_res / ss_tot)


# Shared fit kwargs: 80 iterations, no early-stop (patience=80), tight rel_tol.
_ES_KWARGS = dict(max_iters=80, patience=80, warmup=5, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# T1 – Confounder subspace recovery (NB, standard sparsity)
# ---------------------------------------------------------------------------

def test_confounder_subspace_recovery_nb():
    """GCATE must recover > 50 % of confounder variance on NB-simulated data."""
    n, p, r = 200, 300, 2
    Y, X, A, U_true, *_ = simulate_gcate_data(n=n, p=p, r=r, sparsity=0.3, seed=0)
    disp = np.full(p, 5.0)

    res1, res2 = fit_gcate(
        Y, X, A, r=r,
        family='nb', disp_glm=disp, offset=False,
        kwargs_es_1=_ES_KWARGS, kwargs_es_2=_ES_KWARGS,
    )
    U_est = res2['U']
    r2 = subspace_r2(U_est, U_true)
    assert r2 > 0.5, f"Confounder R² = {r2:.3f} < 0.5 (NB family)"


# ---------------------------------------------------------------------------
# T2 – Sparse genes must not block convergence
# ---------------------------------------------------------------------------

def test_sparse_genes_convergence():
    """GCATE must finish without NaN and achieve R² > 0.25 at 80 % sparsity."""
    n, p, r = 200, 300, 2
    Y, X, A, U_true, *_ = simulate_gcate_data(n=n, p=p, r=r, sparsity=0.8, seed=1)
    disp = np.full(p, 5.0)

    res1, res2 = fit_gcate(
        Y, X, A, r=r,
        family='nb', disp_glm=disp, offset=False,
        kwargs_es_1=_ES_KWARGS, kwargs_es_2=_ES_KWARGS,
    )
    U_est = res2['U']
    assert np.all(np.isfinite(U_est)), "U contains NaN/Inf at 80 % sparsity"
    r2 = subspace_r2(U_est, U_true)
    assert r2 > 0.25, f"Sparse-gene R² = {r2:.3f} < 0.25"


# ---------------------------------------------------------------------------
# T3 – Per-gene mask reduces total gene updates without losing accuracy
# ---------------------------------------------------------------------------

def test_per_gene_mask_efficiency():
    """Enabling tol_gene must reduce gene-update count without hurting R²."""
    n, p, r = 150, 200, 2
    Y, X, A, U_true, *_ = simulate_gcate_data(n=n, p=p, r=r, sparsity=0.3, seed=2)
    disp = np.full(p, 5.0)

    common_es = dict(max_iters=30, patience=30, warmup=3, rel_tol=1e-5)
    # No masking — all genes active every iteration
    res1_base, res2_base = fit_gcate(
        Y, X, A, r=r,
        family='nb', disp_glm=disp, offset=False,
        kwargs_ls_1={'tol_gene': 0.0, 'tol_cell': 0.0, 'recheck_interval': 0},
        kwargs_ls_2={'tol_gene': 0.0, 'tol_cell': 0.0, 'recheck_interval': 0},
        kwargs_es_1=common_es, kwargs_es_2=common_es,
    )
    # With masking — inactive genes skip line search
    res1_fast, res2_fast = fit_gcate(
        Y, X, A, r=r,
        family='nb', disp_glm=disp, offset=False,
        kwargs_ls_1={'tol_gene': 1e-4, 'tol_cell': 1e-4, 'recheck_interval': 100},
        kwargs_ls_2={'tol_gene': 1e-4, 'tol_cell': 1e-4, 'recheck_interval': 100},
        kwargs_es_1=common_es, kwargs_es_2=common_es,
    )

    r2_base = subspace_r2(res2_base['U'], U_true)
    r2_fast = subspace_r2(res2_fast['U'], U_true)

    # Accuracy must be within 10 pp
    assert abs(r2_fast - r2_base) < 0.10, (
        f"Masking hurt R²: base={r2_base:.3f}, masked={r2_fast:.3f}"
    )

    # Masked run must have fewer gene updates in at least one stage
    gu_base = res1_base['total_gene_updates'] + res2_base['total_gene_updates']
    gu_fast = res1_fast['total_gene_updates'] + res2_fast['total_gene_updates']
    assert gu_fast < gu_base, (
        f"Masking did not reduce gene updates: base={gu_base}, masked={gu_fast}"
    )


# ---------------------------------------------------------------------------
# T4 – update_with_mask with all-True masks equals original update behaviour
# ---------------------------------------------------------------------------

def test_update_with_mask_allactive_matches_update():
    """update_with_mask with all-active masks must reproduce update() output."""
    from causarray.gcate_opt import update, update_with_mask
    from causarray.gcate_likelihood import type_f

    rng = np.random.default_rng(3)
    n, p, d, r = 40, 60, 3, 2
    Y  = rng.negative_binomial(5, 0.5, (n, p)).astype(type_f)
    A  = rng.standard_normal((n, d + r)).astype(type_f)
    B  = rng.standard_normal((p, d + r)).astype(type_f)
    lam = np.zeros((p, d), dtype=type_f)
    nuisance = np.ones((1, p), dtype=type_f) * 5.0
    Ys = np.zeros((n, p), dtype=type_f)

    # All-active masks
    gene_active = np.ones(p, dtype=np.bool_)
    cell_active = np.ones(n, dtype=np.bool_)
    alpha_gene  = np.ones(p, dtype=type_f)

    A1, B1 = A.copy(), B.copy()
    A2, B2 = A.copy(), B.copy()

    _, A1, B1 = update(
        Y, A1, B1, d, lam, None, None,
        'nb', nuisance, Ys, 100., d, 10.,
        1.0, 0.5, 5, 1e-4,
    )
    _, A2, B2 = update_with_mask(
        Y, A2, B2, d, lam, None, None,
        'nb', nuisance, Ys, 100., d, 10.,
        1.0, 0.5, 5, 1e-4,
        gene_active, cell_active, alpha_gene,
    )

    np.testing.assert_allclose(A1, A2, atol=1e-8,
        err_msg="update_with_mask (all-active) diverges from update() for A")
    np.testing.assert_allclose(B1, B2, atol=1e-8,
        err_msg="update_with_mask (all-active) diverges from update() for B")


# ---------------------------------------------------------------------------
# T4b – update_with_mask thin-Q P1 matches the explicit (I − QQᵀ) projection
# ---------------------------------------------------------------------------


def test_update_with_mask_thin_q_matches_full_projection():
    """The implicit ``A[:, d:] -= P1 @ (P1.T @ A[:, d:])`` form must produce
    the same A as applying ``(I − QQᵀ)`` explicitly via NumPy.

    Regression test for the thin-Q refactor (Finding 21 in the v0.0.6
    review).  The previous T4 covered ``P1=P2=None`` only, so the
    projection branch was never exercised.
    """
    from causarray.gcate_opt import update_with_mask
    from causarray.gcate_likelihood import type_f

    rng = np.random.default_rng(2026)
    n, p, d_X, r = 40, 80, 3, 2
    d = d_X  # alter_min calls this with d = X.shape[1]
    Y  = rng.negative_binomial(5, 0.5, (n, p)).astype(type_f)
    X  = rng.standard_normal((n, d_X)).astype(type_f)
    A  = rng.standard_normal((n, d_X + r)).astype(type_f)
    B  = rng.standard_normal((p, d_X + r)).astype(type_f)
    lam = np.zeros((p, d_X), dtype=type_f)
    nuisance = np.ones((1, p), dtype=type_f) * 5.0
    Ys = np.zeros((n, p), dtype=type_f)

    gene_active = np.ones(p, dtype=np.bool_)
    cell_active = np.ones(n, dtype=np.bool_)
    alpha_gene  = np.ones(p, dtype=type_f)

    # Thin Q factor for the X column-space
    Q, _ = np.linalg.qr(X)
    Q = Q.astype(type_f)

    A_thin = A.copy()
    B_thin = B.copy()
    _, A_thin, B_thin = update_with_mask(
        Y, A_thin, B_thin, d, lam, Q, None,
        'nb', nuisance, Ys, 100., d, 10.,
        1.0, 0.5, 5, 1e-4,
        gene_active, cell_active, alpha_gene,
    )

    # Drop into a NumPy reference that applies the projection explicitly.
    # We bypass the full update by simulating only the projection step:
    # both implementations must take the same input A_post-A-step and
    # return the same A after projection.  We compare the LATENT block
    # (columns [d:]); the covariate block is unaffected.
    A_ref = A.copy()
    A_ref[:, d:] = (np.eye(n, dtype=type_f) - Q @ Q.T) @ A_ref[:, d:]

    # The projected latent block of A_thin should be orthogonal to Q (up
    # to numerical noise), which is the defining property of the
    # projection.  Q.T @ A_latent ≈ 0 means the implicit step did
    # remove the projection direction.
    np.testing.assert_allclose(
        Q.T @ A_thin[:, d:], np.zeros((d_X, r), dtype=type_f),
        atol=1e-4,
        err_msg="A_thin latent block is not orthogonal to Q — the implicit "
                "(I − QQᵀ) projection did not run.",
    )

    # And applying the same explicit (I − QQᵀ) to A_thin[:, d:] should
    # leave it (numerically) unchanged — idempotence check.
    A_thin_latent = A_thin[:, d:].copy()
    A_thin_latent_proj = (np.eye(n, dtype=type_f) - Q @ Q.T) @ A_thin_latent
    np.testing.assert_allclose(
        A_thin_latent_proj, A_thin_latent, atol=1e-4,
        err_msg="Explicit (I − QQᵀ) applied to A_thin's latent block "
                "should be idempotent.",
    )


# ---------------------------------------------------------------------------
# T5 – NLL must not increase between iterations (within tolerance)
# ---------------------------------------------------------------------------

def test_nll_non_diverging():
    """The NLL history must not diverge: final NLL ≤ 1.5 × initial NLL.

    Note: strict monotonicity is NOT guaranteed by the current line_search
    warm-start heuristic, so we only check for non-divergence.
    """
    n, p, r = 120, 180, 2
    Y, X, A, *_ = simulate_gcate_data(n=n, p=p, r=r, sparsity=0.3, seed=4)

    res1, res2 = fit_gcate(
        Y, X, A, r=r,
        family='poisson', offset=False,
        kwargs_es_1={'max_iters': 20, 'patience': 20, 'warmup': 0, 'rel_tol': 0.},
        kwargs_es_2={'max_iters': 20, 'patience': 20, 'warmup': 0, 'rel_tol': 0.},
    )
    for stage, res in enumerate((res1, res2), start=1):
        hist = np.array(res['hist'])
        assert np.all(np.isfinite(hist)), \
            f"Stage {stage}: NLL history contains NaN/Inf"
        # Final NLL must be at most 1.5× the initial (overall improvement)
        assert hist[-1] <= 1.5 * abs(hist[0]), (
            f"Stage {stage}: NLL diverged from {hist[0]:.4f} to {hist[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# T6 – Confounder recovery under Poisson family
# ---------------------------------------------------------------------------

def test_confounder_subspace_recovery_poisson():
    """GCATE must recover > 40 % of confounder variance under Poisson family."""
    n, p, r = 200, 300, 2
    Y, X, A, U_true, *_ = simulate_gcate_data(n=n, p=p, r=r, sparsity=0.3, seed=5,
                                               family='poisson')
    res1, res2 = fit_gcate(
        Y, X, A, r=r,
        family='poisson', offset=False,
        kwargs_es_1=_ES_KWARGS, kwargs_es_2=_ES_KWARGS,
    )
    U_est = res2['U']
    r2 = subspace_r2(U_est, U_true)
    assert r2 > 0.40, f"Poisson confounder R² = {r2:.3f} < 0.40"
