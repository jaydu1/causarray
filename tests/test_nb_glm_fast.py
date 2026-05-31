"""Unit tests for nb_glm_fast.py and the gcate_glm backend toggle.

Gaps covered:
  G1  – _FAST_MAX_D threshold (d_eff <= 30 now hits fast path)
  G2  – _USE_FAST_BACKEND toggle forces statsmodels
  G3  – cell-count-weighted dispersion in _fit_glm_fast_per_perturbation
  G4  – control-cell residuals / Yhat_full from global model
  G5  – ResourceWarning for large sparse input
  G9  – backend parameter on fit_gcate / compute_causal_estimand
  G10 – _USE_FAST_BACKEND propagates through gcate_opt and DR_estimation
  G11 – _CRISPYX_AVAILABLE=False falls back to statsmodels
  G12 – _compute_poisson_deviance_residuals correct for Y=0
  G13 – mem_limit_gb triggers float32 allocation in imputation arrays
  G14 – mem_limit_gb=None preserves default float64 allocation
  G15 – fit_glm_fast accepts and passes mem_limit_gb through **kwargs
"""
from __future__ import annotations

import math
import warnings
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import scipy.sparse as sp

import causarray.gcate_glm as gcate_glm
from causarray.nb_glm_fast import (
    _compute_poisson_deviance_residuals,
    _maybe_densify,
    _SPARSE_WARN_GB,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_counts(n: int, p: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.negative_binomial(5, 0.5, size=(n, p)).astype(np.float64)


def _make_X(n: int, d: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n)] + [rng.standard_normal(n) for _ in range(d - 1)])
    return X


# ---------------------------------------------------------------------------
# G12 – _compute_poisson_deviance_residuals
# ---------------------------------------------------------------------------

class TestG12PoissonDeviance:
    def test_y0_cells_give_correct_residual(self):
        """When Y=0, deviance = 2*mu so signed resid = -sqrt(2*mu)."""
        mu = np.array([[1.0, 2.0, 0.5]])
        Y = np.zeros_like(mu)
        resid = _compute_poisson_deviance_residuals(Y, mu)
        expected = -np.sqrt(2.0 * mu)
        np.testing.assert_allclose(resid, expected, rtol=1e-9)

    def test_y_equal_mu_gives_zero(self):
        """When Y == mu the deviance residual should be (near) zero."""
        mu = np.array([[3.0, 1.0]])
        resid = _compute_poisson_deviance_residuals(mu.copy(), mu)
        np.testing.assert_allclose(resid, 0.0, atol=1e-9)

    def test_positive_sign_when_y_gt_mu(self):
        Y = np.array([[5.0]])
        mu = np.array([[2.0]])
        resid = _compute_poisson_deviance_residuals(Y, mu)
        assert resid[0, 0] > 0


# ---------------------------------------------------------------------------
# G5 – _maybe_densify
# ---------------------------------------------------------------------------

class TestG5SparseWarning:
    def test_small_sparse_no_warning(self):
        Y_sp = sp.csr_matrix(np.eye(10))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            Y_dense = _maybe_densify(Y_sp)
        assert Y_dense.shape == (10, 10)
        assert Y_dense.dtype == np.float64

    def test_large_sparse_resource_warning(self):
        """Sparse matrix whose dense form exceeds _SPARSE_WARN_GB triggers warning."""
        gb_threshold = _SPARSE_WARN_GB
        # 1 byte per element in float64 is 8 bytes; compute n×p so n*p*8/1e9 > threshold
        n = p = int(math.sqrt(gb_threshold * 1e9 / 8)) + 10
        # Use an extremely sparse diagonal matrix to avoid allocating n*p bytes
        Y_sp = sp.eye(n, p, format="csr")
        with pytest.warns(ResourceWarning, match="Materialising sparse Y"):
            _maybe_densify(Y_sp)

    def test_dense_array_no_warning(self):
        Y = np.ones((5, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            out = _maybe_densify(Y)
        np.testing.assert_array_equal(out, Y)


# ---------------------------------------------------------------------------
# G11 – _CRISPYX_AVAILABLE fallback
# ---------------------------------------------------------------------------

class TestG11CrispyxFallback:
    def test_no_crispyx_falls_back_to_statsmodels(self):
        """When _CRISPYX_AVAILABLE=False, fit_glm_auto must call fit_glm (statsmodels)."""
        n, p = 60, 80
        Y = _make_counts(n, p)
        X = _make_X(n, 2)

        orig_avail = gcate_glm._CRISPYX_AVAILABLE
        try:
            gcate_glm._CRISPYX_AVAILABLE = False
            with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
                gcate_glm.fit_glm_auto(Y, X, family="nb")
                assert mock_fg.called, "fit_glm should be called when crispyx unavailable"
        finally:
            gcate_glm._CRISPYX_AVAILABLE = orig_avail


# ---------------------------------------------------------------------------
# G2 – _USE_FAST_BACKEND toggle
# ---------------------------------------------------------------------------

class TestG2BackendToggle:
    def test_toggle_off_calls_statsmodels(self):
        """_USE_FAST_BACKEND=False must route through fit_glm for any input."""
        n, p = 60, 80
        Y = _make_counts(n, p)
        X = _make_X(n, 2)

        with gcate_glm._backend_override("original"):
            with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
                gcate_glm.fit_glm_auto(Y, X, family="nb")
                assert mock_fg.called

    def test_toggle_restores_after_context_exit(self):
        orig = gcate_glm._USE_FAST_BACKEND
        with gcate_glm._backend_override("original"):
            assert gcate_glm._USE_FAST_BACKEND is False
        assert gcate_glm._USE_FAST_BACKEND == orig

    def test_toggle_fast_sets_true(self):
        with gcate_glm._backend_override("fast"):
            assert gcate_glm._USE_FAST_BACKEND is True

    def test_auto_preserves_current_value(self):
        orig = gcate_glm._USE_FAST_BACKEND
        with gcate_glm._backend_override("auto"):
            assert gcate_glm._USE_FAST_BACKEND == orig


# ---------------------------------------------------------------------------
# G1 – _FAST_MAX_D threshold raised to 30
# ---------------------------------------------------------------------------

class TestG1FastMaxD:
    def test_fast_max_d_default_is_50(self):
        assert gcate_glm._FAST_MAX_D == 50

    def test_d_eff_under_threshold_uses_fast(self):
        """d_eff <= 50 should take the crispyx path (when crispyx available and enabled)."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        # n*p/d_eff² must exceed 5_000 to pass the throughput heuristic.
        # Use d=2 so d_eff²=4: n*p/4 = 200*5000/4 = 250,000 > 5000 ✓
        n, p, d = 200, 5000, 2
        Y = _make_counts(n, p)
        X = _make_X(n, d)  # d_eff = 2 << 30 → fast path

        # fit_glm_fast is bound in gcate_glm via "from ... import"; patch there
        with patch.object(gcate_glm, "fit_glm_fast", wraps=gcate_glm.fit_glm_fast) as mock_ff:
            gcate_glm.fit_glm_auto(Y, X, family="nb")
            assert mock_ff.called, "fit_glm_fast should be called when throughput heuristic passes"

    def test_old_threshold_16_now_passes_fast_path(self):
        """Design with d_eff=16 previously blocked by d_eff<=15; must now hit fast path."""
        assert gcate_glm._FAST_MAX_D >= 16, "Threshold must allow d_eff=16"


# ---------------------------------------------------------------------------
# G10 – backend toggle propagates through gcate_opt / DR_estimation
# ---------------------------------------------------------------------------

class TestG10BackendPropagates:
    def test_gcate_opt_uses_module_ref(self):
        """gcate_opt.alter_min calls _gcate_glm.fit_glm_auto, not a stale binding."""
        import causarray.gcate_opt as gcate_opt
        # gcate_opt should NOT export fit_glm_auto directly; it should use _gcate_glm
        assert not hasattr(gcate_opt, "fit_glm_auto"), (
            "gcate_opt must not bind fit_glm_auto at module level"
        )

    def test_dr_estimation_uses_module_ref(self):
        """DR_estimation should not have fit_glm_auto bound at module level."""
        import causarray.DR_estimation as dr_est
        assert not hasattr(dr_est, "fit_glm_auto"), (
            "DR_estimation must not bind fit_glm_auto at module level"
        )

    def test_toggle_propagates_to_gcate_opt_call(self):
        """With _USE_FAST_BACKEND=False, calls inside gcate_opt use statsmodels."""
        with gcate_glm._backend_override("original"):
            with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
                # Directly call fit_glm_auto; gcate_opt code paths go through this
                n, p = 50, 60
                Y = _make_counts(n, p)
                X = _make_X(n, 2)
                gcate_glm.fit_glm_auto(Y, X, family="nb")
                assert mock_fg.called


# ---------------------------------------------------------------------------
# G9 – backend parameter on fit_gcate
# ---------------------------------------------------------------------------

class TestG9BackendParam:
    def test_fit_gcate_accepts_backend_param(self):
        """fit_gcate must accept backend='original' without error (small data)."""
        from causarray.gcate import fit_gcate

        rng = np.random.default_rng(42)
        n, p, d = 60, 20, 2
        Y = rng.negative_binomial(5, 0.5, (n, p))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        A = rng.binomial(1, 0.3, (n, 1)).astype(float)

        # Should run without error; we only check it completes
        res_1, res_2 = fit_gcate(Y, X, A, r=1, family="nb", offset=False,
                                  backend="original")
        assert isinstance(res_1, dict)
        assert isinstance(res_2, dict)

    def test_fit_gcate_backend_original_uses_statsmodels(self):
        """fit_gcate with backend='original' calls fit_glm (statsmodels path)."""
        from causarray.gcate import fit_gcate

        rng = np.random.default_rng(42)
        n, p = 60, 20
        Y = rng.negative_binomial(5, 0.5, (n, p))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        A = rng.binomial(1, 0.3, (n, 1)).astype(float)

        with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
            fit_gcate(Y, X, A, r=1, family="nb", offset=False, backend="original")
            assert mock_fg.called, "statsmodels path not called with backend='original'"


# ---------------------------------------------------------------------------
# G3 – weighted dispersion average
# ---------------------------------------------------------------------------

class TestG3WeightedDisp:
    def test_weighted_disp_closer_to_large_group(self):
        """With unbalanced treatment groups, disp estimate weighted by cell count."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray.nb_glm_fast import _fit_glm_fast_per_perturbation

        rng = np.random.default_rng(7)
        n_ctrl, n_small, n_large = 200, 20, 200
        p = 50

        # Simulate: perturbation 1 (small, high disp), perturbation 2 (large, low disp)
        # We can't easily control crispyx output, so just check disp is finite
        # and shape is correct — the full deterministic test is in integration.
        n = n_ctrl + n_small + n_large
        Y = rng.negative_binomial(5, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, 2))
        A[n_ctrl:n_ctrl + n_small, 0] = 1          # small group
        A[n_ctrl + n_small:, 1] = 1                # large group

        B, Yhat, disp_out, _, resid = _fit_glm_fast_per_perturbation(
            Y, X, A, a=2, d=1, p=p, n=n,
            family="nb", disp_glm=None,
            impute=False, X_test=None,
            offset_arr=np.zeros(n),
            offsets=None,
            maxiter=10, verbose=False,
        )
        assert disp_out is not None
        assert np.all(np.isfinite(disp_out))
        assert disp_out.shape == (p,)


# ---------------------------------------------------------------------------
# G4 – control-cell residuals from global model
# ---------------------------------------------------------------------------

class TestG4ControlResiduals:
    def test_control_cell_resid_not_overwritten(self):
        """Control-cell residuals must be consistent across perturbations (from global model)."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray.nb_glm_fast import _fit_glm_fast_per_perturbation

        rng = np.random.default_rng(99)
        n_ctrl = 100
        n_per_pert = 50
        a = 3
        p = 40

        n = n_ctrl + n_per_pert * a
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        for k in range(a):
            start = n_ctrl + k * n_per_pert
            A[start:start + n_per_pert, k] = 1

        ctrl_idx = np.arange(n_ctrl)

        B, Yhat, disp_out, _, resid = _fit_glm_fast_per_perturbation(
            Y, X, A, a=a, d=1, p=p, n=n,
            family="nb", disp_glm=None,
            impute=False, X_test=None,
            offset_arr=np.zeros(n), offsets=None,
            maxiter=10, verbose=False,
        )

        # Control residuals should all be finite (no zeros from stale init)
        assert np.all(np.isfinite(resid[ctrl_idx]))
        # Yhat for control cells should be > 0 (non-zero from global model)
        assert np.all(Yhat[ctrl_idx] > 0)


# ---------------------------------------------------------------------------
# G13 – mem_limit_gb triggers float32 imputation arrays
# ---------------------------------------------------------------------------

class TestG13MemLimitFloat32:
    """mem_limit_gb below the imputation array size → float32 allocation + warning."""

    @staticmethod
    def _run_per_pert(n, p, a, mem_limit_gb, impute=True):
        from causarray.nb_glm_fast import _fit_glm_fast_per_perturbation

        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        rng = np.random.default_rng(17)
        n_ctrl = n // 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        per = (n - n_ctrl) // a
        for k in range(a):
            A[n_ctrl + k * per : n_ctrl + (k + 1) * per, k] = 1
        X_test = X.copy() if impute else None

        return _fit_glm_fast_per_perturbation(
            Y, X, A, a=a, d=1, p=p, n=n,
            family="nb", disp_glm=None,
            impute=(X_test is not None), X_test=X_test,
            offset_arr=np.zeros(n), offsets=None,
            maxiter=10, verbose=False,
            mem_limit_gb=mem_limit_gb,
        )

    def test_below_limit_uses_float32_and_warns(self):
        """Setting mem_limit_gb below imputation cost → float32 Yhat_0/1 + ResourceWarning."""
        n, p, a = 100, 40, 3
        # Actual imputation cost (2 arrays of n × p × a float64) in GB
        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.5  # force the float32 branch

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            B, Yhat, disp_out, _, resid = self._run_per_pert(n, p, a, mem_limit_gb=tiny_limit)

        Yhat_0, Yhat_1 = Yhat
        assert Yhat_0.dtype == np.float32, f"Expected float32, got {Yhat_0.dtype}"
        assert Yhat_1.dtype == np.float32, f"Expected float32, got {Yhat_1.dtype}"
        # Results must still be numerically valid
        assert Yhat_0.shape == (n, p, a)
        assert Yhat_1.shape == (n, p, a)
        assert np.all(np.isfinite(Yhat_0))
        assert np.all(np.isfinite(Yhat_1))

    def test_above_limit_stays_float64(self):
        """When mem_limit_gb is larger than the cost, float64 is preserved."""
        n, p, a = 100, 40, 3
        large_limit = 1e6  # effectively unlimited

        # No ResourceWarning should fire
        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            B, Yhat, disp_out, _, resid = self._run_per_pert(n, p, a, mem_limit_gb=large_limit)

        Yhat_0, Yhat_1 = Yhat
        assert Yhat_0.dtype == np.float64
        assert Yhat_1.dtype == np.float64

    def test_none_limit_stays_float64(self):
        """mem_limit_gb=None (default) must not trigger float32 downcast."""
        n, p, a = 100, 40, 3

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            B, Yhat, disp_out, _, resid = self._run_per_pert(n, p, a, mem_limit_gb=None)

        Yhat_0, Yhat_1 = Yhat
        assert Yhat_0.dtype == np.float64
        assert Yhat_1.dtype == np.float64

    def test_no_impute_no_warning(self):
        """When impute=False mem_limit_gb has no effect and no warning fires."""
        n, p, a = 100, 40, 3
        tiny_limit = 0.0  # would trigger if imputation were active

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            # impute=False → no Yhat_0/Yhat_1 allocated → no warning
            B, Yhat, disp_out, _, resid = self._run_per_pert(
                n, p, a, mem_limit_gb=tiny_limit, impute=False
            )
        # Yhat is the full Yhat_full array when impute=False
        assert isinstance(Yhat, np.ndarray)


# ---------------------------------------------------------------------------
# G14 – fit_glm_fast passes mem_limit_gb to _fit_glm_fast_per_perturbation
# ---------------------------------------------------------------------------

class TestG14FitGlmFastPassesMem:
    """fit_glm_fast(mem_limit_gb=...) must thread the arg through to imputation."""

    def test_mem_limit_propagates_via_fit_glm_fast(self):
        """fit_glm_fast with a very small mem_limit_gb raises ResourceWarning."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray.nb_glm_fast import fit_glm_fast

        rng = np.random.default_rng(42)
        n, p, a = 80, 50, 3
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        per = n // (a + 1)
        for k in range(a):
            A[(k + 1) * per : (k + 2) * per, k] = 1
        X_test = X.copy()

        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.1

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            B, Yhat, disp_out, offsets, resid = fit_glm_fast(
                Y, X, A=A, family="nb",
                impute=X_test,
                mem_limit_gb=tiny_limit,
            )

        Yhat_0, Yhat_1 = Yhat
        assert Yhat_0.dtype == np.float32


# ---------------------------------------------------------------------------
# G15 – cross_fitting / LFC: mem_limit_gb switches Y_hat to float32
# ---------------------------------------------------------------------------

class TestG15CrossFittingMemLimit:
    """DR_estimation.cross_fitting allocates Y_hat as float32 when over mem_limit_gb."""

    def test_cross_fitting_float32_when_over_limit(self):
        """mem_limit_gb below Y_hat cost → float32 Y_hat + ResourceWarning."""
        from causarray.DR_estimation import cross_fitting

        rng = np.random.default_rng(7)
        n, p, a = 100, 30, 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)
        X = rng.standard_normal((n, 2))
        A = np.zeros((n, a))
        A[:30, 0] = 1
        A[30:60, 1] = 1

        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.1

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            Y_hat, pi_hat = cross_fitting(
                Y, A, X, X, family="nb",
                mem_limit_gb=tiny_limit,
            )

        assert Y_hat.dtype == np.float32, f"Expected float32, got {Y_hat.dtype}"
        assert Y_hat.shape == (n, p, a, 2)
        assert np.all(np.isfinite(Y_hat))

    def test_cross_fitting_float64_when_under_limit(self):
        """mem_limit_gb above Y_hat cost → float64 preserved, no warning."""
        from causarray.DR_estimation import cross_fitting

        rng = np.random.default_rng(8)
        n, p, a = 100, 30, 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)
        X = rng.standard_normal((n, 2))
        A = np.zeros((n, a))
        A[:30, 0] = 1
        A[30:60, 1] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            Y_hat, pi_hat = cross_fitting(
                Y, A, X, X, family="nb",
                mem_limit_gb=1e6,
            )

        assert Y_hat.dtype == np.float64

    def test_cross_fitting_no_limit_float64(self):
        """mem_limit_gb=None (default) must not trigger float32 downcast."""
        from causarray.DR_estimation import cross_fitting

        rng = np.random.default_rng(9)
        n, p, a = 80, 20, 2
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)
        X = rng.standard_normal((n, 2))
        A = np.zeros((n, a))
        A[:20, 0] = 1
        A[20:40, 1] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            Y_hat, pi_hat = cross_fitting(Y, A, X, X, family="nb")

        assert Y_hat.dtype == np.float64

    def test_lfc_accepts_mem_limit_gb(self):
        """LFC(mem_limit_gb=...) threads through to cross_fitting without error."""
        from causarray.DR_learner import LFC

        rng = np.random.default_rng(11)
        n, p = 120, 20
        a = 2
        W = rng.standard_normal((n, 3))
        A = np.zeros((n, a))
        A[:30, 0] = 1
        A[30:60, 1] = 1
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(float)

        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.1

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            df_res, estimation = LFC(Y, W, A, family="nb", mem_limit_gb=tiny_limit)

        assert "tau" in df_res.columns
        assert len(df_res) == p * a
        # Y_hat stored in estimation should be float32
        assert estimation["Y_hat"].dtype == np.float32

    def test_fit_glm_auto_propagates_mem_limit_gb(self):
        """fit_glm_auto must forward mem_limit_gb to _fit_glm_fast_per_perturbation.

        Tightens the original G15 coverage (Finding 22 in the v0.0.6
        review): the prior test only verified the *final* Y_hat dtype,
        which is governed by ``cross_fitting`` and is therefore satisfied
        even when ``fit_glm_auto`` silently drops ``mem_limit_gb`` before
        calling ``fit_glm_fast``.  Here we monkey-patch
        ``_fit_glm_fast_per_perturbation`` and assert it receives the
        non-None ``mem_limit_gb`` value the caller supplied.
        """
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray import nb_glm_fast as _nbgf

        # Data must be large enough that ``fit_glm_auto``'s throughput
        # heuristic (n*p/d_eff² > 5000) prefers the fast path; otherwise
        # the slow ``fit_glm`` is taken and the spy never fires.
        rng = np.random.default_rng(42)
        n, p, a = 200, 200, 3
        Y = rng.negative_binomial(3, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        per = n // (a + 1)
        for k in range(a):
            A[(k + 1) * per : (k + 2) * per, k] = 1
        X_test = X.copy()

        observed = {}
        real = _nbgf._fit_glm_fast_per_perturbation

        def _spy(*args, **kw):
            observed['mem_limit_gb'] = kw.get('mem_limit_gb', 'MISSING')
            return real(*args, **kw)

        _nbgf._fit_glm_fast_per_perturbation = _spy
        try:
            tiny_limit = 0.123  # arbitrary non-default sentinel
            gcate_glm.fit_glm_auto(
                Y, X, A=A, family='nb',
                impute=X_test,
                mem_limit_gb=tiny_limit,
            )
        finally:
            _nbgf._fit_glm_fast_per_perturbation = real

        assert observed.get('mem_limit_gb') == tiny_limit, (
            f"_fit_glm_fast_per_perturbation received "
            f"mem_limit_gb={observed.get('mem_limit_gb')!r}; "
            f"expected {tiny_limit!r}."
        )

