"""
Test deconfounding performance of GCATE.

Simulates data with unmeasured confounders to verify that GCATE
correctly identifies and adjusts for them, producing more accurate
treatment effect estimates than a naive approach without deconfounding.
"""

import numpy as np
import pytest
from causarray.gcate import fit_gcate
from causarray.DR_learner import LFC


@pytest.fixture
def sim_confounded_data():
    """Simulate NB count data with unmeasured confounders.

    Data generating process:
     - U (n x r): latent confounders drawn from N(0, 1)
     - X_obs: observed covariates
     - A: binary treatment influenced by U  (confounding!)
     - Y ~ NB(mu, disp), where log(mu) depends on X_obs, A, and U
     - true treatment effect (tau) is set per gene
    """
    np.random.seed(2024)
    n, p, r = 300, 30, 2

    # Observed covariates
    X_obs = np.random.randn(n, 2)

    # Latent confounders
    U = np.random.randn(n, r)

    # Confounding: treatment depends on U
    logit_a = 0.8 * U[:, 0] + 0.6 * U[:, 1]
    prob_a = 1.0 / (1.0 + np.exp(-logit_a))
    A = np.random.binomial(1, prob_a, n).astype(float)

    # True treatment effects (half zero, half nonzero)
    tau_true = np.zeros(p)
    tau_true[: p // 2] = np.random.choice([-1, 1], p // 2) * np.random.uniform(0.5, 1.5, p // 2)

    # Coefficients for confounders (large effect → strong confounding)
    gamma = np.random.randn(r, p) * 1.5

    # Gene-level baseline
    beta0 = np.random.uniform(2.0, 3.5, p)

    # Build log-mean
    log_mu = (
        beta0[None, :]
        + X_obs @ np.random.randn(2, p) * 0.2
        + A[:, None] * tau_true[None, :]
        + U @ gamma                       # confounding signal
    )
    mu = np.exp(np.clip(log_mu, -10, 10))

    # Dispersion (moderate overdispersion)
    disp = np.random.uniform(0.5, 2.0, p)
    Y = np.random.negative_binomial(
        n=np.maximum(1.0 / disp, 0.01),
        p=np.clip(1.0 / (1.0 + mu * disp[None, :]), 1e-10, 1 - 1e-10),
    ).astype(float)

    return Y, X_obs, A, tau_true, r


class TestDeconfoundingPerformance:
    """Verify GCATE deconfounding improves treatment effect estimation."""

    def test_gcate_reduces_confounding_bias(self, sim_confounded_data):
        """GCATE-adjusted LFC should be closer to truth than naive LFC.

        Follows the real pipeline:
          1. Naive: LFC(Y, X_obs, A) — no latent factors
          2. GCATE: fit_gcate → extract U → LFC(Y, [X_obs, U], A, [X_obs, U])
        """
        Y, X_obs, A, tau_true, r = sim_confounded_data

        # ---- Naive: LFC without deconfounding ----
        df_naive, _ = LFC(Y, X_obs, A[:, None], family='nb', offset=True)
        lfc_naive = df_naive['tau'].values

        # ---- GCATE: estimate latent factors, then run LFC with them ----
        res_1, res_2 = fit_gcate(
            Y, X_obs, A[:, None], r=r, family='nb', offset=True,
        )
        U_hat = res_2['U']  # estimated latent confounders
        offsets = np.log(res_2['kwargs_glm']['size_factor'])

        df_deconf, _ = LFC(
            Y, np.c_[X_obs, U_hat], A[:, None],
            W_A=np.c_[X_obs, U_hat],
            family='nb', offset=offsets,
        )
        lfc_deconf = df_deconf['tau'].values

        # ---- Compare ----
        corr_naive = np.corrcoef(lfc_naive, tau_true)[0, 1]
        corr_deconf = np.corrcoef(lfc_deconf, tau_true)[0, 1]

        mse_naive = np.mean((lfc_naive - tau_true) ** 2)
        mse_deconf = np.mean((lfc_deconf - tau_true) ** 2)

        print(f"\nNaive  LFC:    corr={corr_naive:.4f}, MSE={mse_naive:.4f}")
        print(f"Deconf (r={r}): corr={corr_deconf:.4f}, MSE={mse_deconf:.4f}")

        # Deconfounding should improve correlation with truth
        assert corr_deconf > corr_naive, (
            f"Deconfounding did not improve LFC: "
            f"corr_deconf={corr_deconf:.4f} <= corr_naive={corr_naive:.4f}"
        )

    def test_gcate_latent_factors_recovered(self, sim_confounded_data):
        """Estimated latent factors should capture the confounding signal."""
        Y, X_obs, A, _, r = sim_confounded_data

        _, res_2 = fit_gcate(
            Y, X_obs, A[:, None], r=r, family='nb', offset=True,
        )

        U_hat = res_2['U']
        assert U_hat.shape == (Y.shape[0], r), (
            f"Expected latent factor shape ({Y.shape[0]}, {r}), got {U_hat.shape}"
        )
        # Latent factor columns should have non-trivial variance
        for k in range(r):
            assert np.std(U_hat[:, k]) > 0.01, (
                f"Latent factor {k} has near-zero variance"
            )
