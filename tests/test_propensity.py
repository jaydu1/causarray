import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss

from causarray import (
    estimate_propensity_scores,
    plot_propensity_scores,
    summarize_propensity_scores,
)
from causarray.DR_estimation import cross_fitting


def test_intercept_only_scores_respect_class_weight():
    A = np.zeros((100, 2))
    A[60:80, 0] = 1
    A[80:100, 1] = 1

    balanced = estimate_propensity_scores(A, np.ones((100, 1)))
    calibrated = estimate_propensity_scores(
        A, np.ones((100, 1)), class_weight=None)

    np.testing.assert_allclose(balanced, 0.5)
    np.testing.assert_allclose(calibrated[:, 0], 0.25)
    np.testing.assert_allclose(calibrated[:, 1], 0.25)


def test_calibrated_option_improves_probability_calibration():
    rng = np.random.default_rng(12)
    n = 5000
    z = rng.standard_normal(n)
    true_pi = 1 / (1 + np.exp(-(-2.0 + 0.8 * z)))
    A = rng.binomial(1, true_pi)
    X = np.c_[np.ones(n), z]

    balanced = estimate_propensity_scores(A, X)
    calibrated = estimate_propensity_scores(A, X, class_weight=None)

    assert abs(calibrated.mean() - A.mean()) < 0.02
    assert abs(balanced.mean() - A.mean()) > 0.15
    assert brier_score_loss(A, calibrated[:, 0]) < brier_score_loss(A, balanced[:, 0])


def test_cross_fitted_scores_are_reproducible_and_unclipped():
    rng = np.random.default_rng(9)
    X = np.c_[np.ones(300), rng.standard_normal((300, 2))]
    A = rng.binomial(1, 1 / (1 + np.exp(-X[:, 1])))

    pi_1 = estimate_propensity_scores(A, X, K=5, random_state=4)
    pi_2 = estimate_propensity_scores(A, X, K=5, random_state=4)

    np.testing.assert_allclose(pi_1, pi_2)
    assert np.all((pi_1 > 0) & (pi_1 < 1))


def test_fold_specific_mask_uses_training_rows():
    rng = np.random.default_rng(5)
    n = 240
    X = np.c_[np.ones(n), rng.standard_normal(n)]
    A = np.zeros((n, 2))
    A[80:120, 0] = 1
    A[120:160, 1] = 1
    mask = np.zeros_like(A, dtype=bool)
    controls = np.arange(n) < 80
    mask[:, 0] = controls | (A[:, 0] == 1)
    mask[:, 1] = controls | (A[:, 1] == 1)

    pi = estimate_propensity_scores(A, X, K=3, mask=mask)

    assert pi.shape == A.shape
    assert np.isfinite(pi).all()


def test_clipping_is_optional_and_applied_after_prediction():
    rng = np.random.default_rng(10)
    z = np.r_[rng.normal(-3, 0.2, 100), rng.normal(3, 0.2, 100)]
    A = np.r_[np.zeros(100), np.ones(100)]
    X = np.c_[np.ones(200), z]

    raw = estimate_propensity_scores(A, X, clip=None, C=100)
    clipped = estimate_propensity_scores(A, X, clip=(0.1, 0.9), C=100)

    assert raw.min() < 0.1 and raw.max() > 0.9
    assert clipped.min() >= 0.1 and clipped.max() <= 0.9


def test_cross_fitting_can_return_raw_and_clipped_scores():
    rng = np.random.default_rng(3)
    n, p = 120, 4
    z = rng.standard_normal(n)
    A = rng.binomial(1, 1 / (1 + np.exp(-3 * z)))[:, None]
    X = np.c_[np.ones(n), z]
    Y = rng.poisson(2, (n, p)).astype(float)
    Y_hat = np.ones((n, p, 1, 2))

    _, clipped, raw = cross_fitting(
        Y, A, X, X, K=3, Y_hat=Y_hat, ps_clip=(0.1, 0.9),
        return_raw_pi=True,
    )

    assert raw.min() < clipped.min() or raw.max() > clipped.max()
    assert clipped.min() >= 0.1 and clipped.max() <= 0.9


def test_cross_fitting_preserves_balanced_default_and_allows_calibrated_scores():
    rng = np.random.default_rng(31)
    n, p = 300, 3
    z = rng.standard_normal(n)
    A = rng.binomial(1, 1 / (1 + np.exp(-(-2 + z))))[:, None]
    X = np.c_[np.ones(n), z]
    Y = rng.poisson(2, (n, p)).astype(float)
    Y_hat = np.ones((n, p, 1, 2))

    _, _, legacy = cross_fitting(
        Y, A, X, X, Y_hat=Y_hat, return_raw_pi=True,
    )
    _, _, calibrated = cross_fitting(
        Y, A, X, X, Y_hat=Y_hat, return_raw_pi=True,
        ps_class_weight=None,
    )
    expected_calibrated = estimate_propensity_scores(A, X, class_weight=None)
    expected_legacy = estimate_propensity_scores(A, X)

    np.testing.assert_allclose(calibrated, expected_calibrated)
    np.testing.assert_allclose(legacy, expected_legacy)
    assert not np.allclose(calibrated, legacy)


def test_deprecated_class_weight_overrides_default():
    rng = np.random.default_rng(32)
    n, p = 200, 2
    z = rng.standard_normal(n)
    A = rng.binomial(1, 1 / (1 + np.exp(-(-2 + z))))[:, None]
    X = np.c_[np.ones(n), z]
    Y = rng.poisson(2, (n, p)).astype(float)
    Y_hat = np.ones((n, p, 1, 2))

    with pytest.warns(FutureWarning, match='ps_class_weight'):
        _, _, raw = cross_fitting(
            Y, A, X, X, Y_hat=Y_hat, return_raw_pi=True,
            class_weight=None,
        )

    np.testing.assert_allclose(
        raw, estimate_propensity_scores(A, X, class_weight=None))


def test_propensity_summary_and_plot_for_named_treatments():
    rng = np.random.default_rng(14)
    A = pd.DataFrame(np.zeros((180, 2)), columns=['pert_a', 'pert_b'])
    A.loc[60:99, 'pert_a'] = 1
    A.loc[100:139, 'pert_b'] = 1
    pi = np.clip(0.2 + 0.1 * rng.standard_normal(A.shape), 0.01, 0.99)

    summary = summarize_propensity_scores(A, pi)
    fig, axes, plotted = plot_propensity_scores(A, pi, treatments=['pert_b'])

    expected = {
        'treatment', 'overlap_ratio', 'outside_overlap_fraction',
        'ess_control_fraction', 'ess_treated_fraction', 'auc', 'brier_score',
    }
    assert expected.issubset(summary.columns)
    assert summary['overlap_ratio'].between(0, 1).all()
    assert summary['ess_treated_fraction'].between(0, 1).all()
    assert plotted.equals(summary)
    assert axes.size >= 1
    plt.close(fig)
