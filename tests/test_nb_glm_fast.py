"""Unit tests for nb_glm_fast.py and the gcate_glm backend toggle."""
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
# Poisson deviance residuals
# ---------------------------------------------------------------------------

class TestPoissonDevianceResiduals:
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
# Sparse densification
# ---------------------------------------------------------------------------

class TestSparseDensification:
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
        n = p = int(math.sqrt(gb_threshold * 1e9 / 8)) + 10
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
# Crispyx availability fallback
# ---------------------------------------------------------------------------

class TestCrispyxFallback:
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
# Fast-path threshold (_FAST_MAX_D)
# ---------------------------------------------------------------------------

class TestFastMaxDThreshold:
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
        X = _make_X(n, d)

        with patch.object(gcate_glm, "fit_glm_fast", wraps=gcate_glm.fit_glm_fast) as mock_ff:
            gcate_glm.fit_glm_auto(Y, X, family="nb")
            assert mock_ff.called, "fit_glm_fast should be called when throughput heuristic passes"

    def test_old_threshold_16_now_passes_fast_path(self):
        """Design with d_eff=16 previously blocked by d_eff<=15; must now hit fast path."""
        assert gcate_glm._FAST_MAX_D >= 16, "Threshold must allow d_eff=16"


# ---------------------------------------------------------------------------
# Backend toggle, backend param, backend propagation
# ---------------------------------------------------------------------------

class TestBackendToggle:
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


class TestBackendParam:
    def test_fit_gcate_accepts_backend_param(self):
        """fit_gcate must accept backend='original' without error (small data)."""
        from causarray.gcate import fit_gcate

        rng = np.random.default_rng(42)
        n, p, d = 60, 20, 2
        Y = rng.negative_binomial(5, 0.5, (n, p))
        X = np.column_stack([np.ones(n), rng.standard_normal(n)])
        A = rng.binomial(1, 0.3, (n, 1)).astype(float)

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


class TestBackendPropagation:
    def test_gcate_opt_uses_module_ref(self):
        """gcate_opt.alter_min calls _gcate_glm.fit_glm_auto, not a stale binding."""
        import causarray.gcate_opt as gcate_opt
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
                n, p = 50, 60
                Y = _make_counts(n, p)
                X = _make_X(n, 2)
                gcate_glm.fit_glm_auto(Y, X, family="nb")
                assert mock_fg.called


# ---------------------------------------------------------------------------
# Weighted dispersion average
# ---------------------------------------------------------------------------

class TestWeightedDispersion:
    def test_weighted_disp_closer_to_large_group(self):
        """With unbalanced treatment groups, disp estimate weighted by cell count."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray.nb_glm_fast import _fit_glm_fast_per_perturbation

        rng = np.random.default_rng(7)
        n_ctrl, n_small, n_large = 200, 20, 200
        p = 50

        n = n_ctrl + n_small + n_large
        Y = rng.negative_binomial(5, 0.5, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, 2))
        A[n_ctrl:n_ctrl + n_small, 0] = 1
        A[n_ctrl + n_small:, 1] = 1

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
# Control-cell residuals from global model
# ---------------------------------------------------------------------------

class TestControlCellResiduals:
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

        assert np.all(np.isfinite(resid[ctrl_idx]))
        assert np.all(Yhat[ctrl_idx] > 0)


# ---------------------------------------------------------------------------
# Per-perturbation Stage-2 GLM loop
# ---------------------------------------------------------------------------

class TestPerPerturbationStage2:
    def test_multi_treatment_uses_one_stage2_fit_per_perturbation(self):
        """Multi-column A should use the historical per-perturbation GLM loop."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray.nb_glm_fast import _fit_glm_fast_per_perturbation
        from crispyx.glm import NBGLMBatchFitter

        rng = np.random.default_rng(123)
        n, p, a = 72, 10, 3
        Y = rng.poisson(2.0, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        for k in range(a):
            A[18 + k * 12 : 18 + (k + 1) * 12, k] = 1

        real = NBGLMBatchFitter.fit_batch_with_joint_offsets

        def _spy(self, *args, **kwargs):
            return real(self, *args, **kwargs)

        with patch.object(
            NBGLMBatchFitter,
            "fit_batch_with_joint_offsets",
            autospec=True,
            side_effect=_spy,
        ) as mock_fit:
            _fit_glm_fast_per_perturbation(
                Y, X, A, a=a, d=1, p=p, n=n,
                family="poisson", disp_glm=None,
                impute=False, X_test=None,
                offset_arr=np.zeros(n),
                offsets=None,
                maxiter=3, verbose=False,
            )

        assert mock_fit.call_count == a


# ---------------------------------------------------------------------------
# Stage-1 subsample random_state inheritance
# ---------------------------------------------------------------------------

class TestStage1RandomState:
    def test_fit_glm_fast_passes_random_state_to_per_perturbation(self):
        """Public fit_glm_fast must thread random_state into the multi-A path."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray import nb_glm_fast as _nbgf

        rng = np.random.default_rng(456)
        n, p, a = 40, 5, 2
        Y = rng.poisson(2.0, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        A[10:20, 0] = 1
        A[20:30, 1] = 1

        observed = {}

        def _stub(*args, **kwargs):
            observed["random_state"] = kwargs.get("random_state")
            _, _, _, a_arg, d_arg, p_arg, n_arg = args[:7]
            return (
                np.zeros((p_arg, d_arg + a_arg)),
                np.zeros((n_arg, p_arg)),
                None,
                None,
                np.zeros((n_arg, p_arg)),
            )

        with patch.object(_nbgf, "_fit_glm_fast_per_perturbation", side_effect=_stub):
            _nbgf.fit_glm_fast(
                Y, X, A=A, family="poisson", random_state=123,
            )

        assert observed["random_state"] == 123

    def test_stage1_subsample_uses_caller_random_state(self):
        """For n > 3000, Stage-1 sampling must seed from random_state."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray.nb_glm_fast import _fit_glm_fast_per_perturbation

        rng = np.random.default_rng(321)
        n, p, a = 3001, 2, 2
        Y = rng.poisson(2.0, (n, p)).astype(np.float64)
        X = np.ones((n, 1))
        A = np.zeros((n, a))
        A[1000:2000, 0] = 1
        A[2000:, 1] = 1

        with patch("numpy.random.default_rng", wraps=np.random.default_rng) as mock_rng:
            _fit_glm_fast_per_perturbation(
                Y, X, A, a=a, d=1, p=p, n=n,
                family="poisson", disp_glm=None,
                impute=False, X_test=None,
                offset_arr=np.zeros(n),
                offsets=None,
                maxiter=1, verbose=False,
                random_state=123,
            )

        assert any(call.args == (123,) for call in mock_rng.call_args_list)


# ---------------------------------------------------------------------------
# Memory limits — per-perturbation imputation arrays
# ---------------------------------------------------------------------------

class TestMemoryLimitPerPerturbation:
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
        true_gb = n * p * a * 2 * 8 / 1e9
        tiny_limit = true_gb * 0.5

        with pytest.warns(ResourceWarning, match="mem_limit_gb"):
            B, Yhat, disp_out, _, resid = self._run_per_pert(n, p, a, mem_limit_gb=tiny_limit)

        Yhat_0, Yhat_1 = Yhat
        assert Yhat_0.dtype == np.float32, f"Expected float32, got {Yhat_0.dtype}"
        assert Yhat_1.dtype == np.float32, f"Expected float32, got {Yhat_1.dtype}"
        assert Yhat_0.shape == (n, p, a)
        assert Yhat_1.shape == (n, p, a)
        assert np.all(np.isfinite(Yhat_0))
        assert np.all(np.isfinite(Yhat_1))

    def test_above_limit_stays_float64(self):
        """When mem_limit_gb is larger than the cost, float64 is preserved."""
        n, p, a = 100, 40, 3
        large_limit = 1e6

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
        tiny_limit = 0.0

        with warnings.catch_warnings():
            warnings.simplefilter("error", ResourceWarning)
            B, Yhat, disp_out, _, resid = self._run_per_pert(
                n, p, a, mem_limit_gb=tiny_limit, impute=False
            )
        assert isinstance(Yhat, np.ndarray)


# ---------------------------------------------------------------------------
# Memory limits — fit_glm_fast passthrough
# ---------------------------------------------------------------------------

class TestMemoryLimitPassthrough:
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
# Memory limits — cross_fitting and LFC
# ---------------------------------------------------------------------------

class TestMemoryLimitCrossFitting:
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
        assert estimation["Y_hat"].dtype == np.float32

    def test_fit_glm_auto_propagates_mem_limit_gb(self):
        """fit_glm_auto must forward mem_limit_gb to _fit_glm_fast_per_perturbation."""
        if not gcate_glm._CRISPYX_AVAILABLE:
            pytest.skip("crispyx not installed")

        from causarray import nb_glm_fast as _nbgf

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
            tiny_limit = 0.123
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
