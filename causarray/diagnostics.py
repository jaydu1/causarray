"""Diagnostics for treatment overlap and propensity-score quality."""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


def _coerce_treatment_inputs(A, pi_hat, treatment_names=None):
    if isinstance(A, pd.DataFrame):
        if treatment_names is None:
            treatment_names = list(A.columns)
        A = A.values
    else:
        A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]

    pi_hat = np.asarray(pi_hat, dtype=float)
    if pi_hat.ndim == 1:
        pi_hat = pi_hat[:, None]
    if A.shape != pi_hat.shape:
        raise ValueError('A and pi_hat must have the same shape')
    if not np.all(np.isin(A, (0, 1))):
        raise ValueError('A must contain only binary treatment indicators')
    if np.any(~np.isfinite(pi_hat)) or np.any((pi_hat < 0) | (pi_hat > 1)):
        raise ValueError('pi_hat must contain finite probabilities in [0, 1]')

    if treatment_names is None:
        treatment_names = list(range(A.shape[1]))
    if len(treatment_names) != A.shape[1]:
        raise ValueError('treatment_names must match the number of treatments')
    return A, pi_hat, list(treatment_names)


def _effective_sample_size(weights):
    weights = np.asarray(weights, dtype=float)
    denom = np.sum(weights ** 2)
    return float(np.sum(weights) ** 2 / denom) if denom > 0 else np.nan


def summarize_propensity_scores(
    A, pi_hat, treatment_names=None, overlap_bounds=(0.05, 0.95),
    clip_bounds=(0.01, 0.99), bins=40,
):
    """Summarize overlap and inverse-weight stability for each treatment.

    Other perturbations are excluded from a treatment's diagnostic comparison;
    each row compares that treatment with shared all-zero controls.
    """
    A, pi_hat, treatment_names = _coerce_treatment_inputs(
        A, pi_hat, treatment_names)
    lo, hi = overlap_bounds
    if not 0 <= lo < hi <= 1:
        raise ValueError('overlap_bounds must satisfy 0 <= lower < upper <= 1')
    if bins < 2:
        raise ValueError('bins must be at least 2')

    ctrl = np.sum(A, axis=1) == 0
    if not np.any(ctrl):
        raise ValueError('At least one all-zero control row is required')
    rows = []
    for j, name in enumerate(treatment_names):
        case = A[:, j] == 1
        if not np.any(case):
            raise ValueError(f'Treatment {name} has no treated rows')
        eligible = ctrl | case
        y = A[eligible, j].astype(int)
        p = pi_hat[eligible, j]
        p_ctrl, p_case = p[y == 0], p[y == 1]

        h_ctrl, edges = np.histogram(p_ctrl, bins=bins, range=(0, 1))
        h_case, _ = np.histogram(p_case, bins=edges)
        h_ctrl = h_ctrl / h_ctrl.sum() if h_ctrl.sum() else h_ctrl
        h_case = h_case / h_case.sum() if h_case.sum() else h_case
        overlap_ratio = float(np.minimum(h_ctrl, h_case).sum())

        eps = np.finfo(float).eps
        ess_ctrl = _effective_sample_size(1 / np.clip(1 - p_ctrl, eps, None))
        ess_case = _effective_sample_size(1 / np.clip(p_case, eps, None))
        clipped = np.zeros(p.shape, dtype=bool)
        if clip_bounds is not None:
            clipped = np.isclose(p, clip_bounds[0]) | np.isclose(p, clip_bounds[1])

        rows.append({
            'treatment': name,
            'n_control': int((y == 0).sum()),
            'n_treated': int((y == 1).sum()),
            'prevalence': float(y.mean()),
            'overlap_ratio': overlap_ratio,
            'auc': float(roc_auc_score(y, p)),
            'brier_score': float(brier_score_loss(y, p)),
            'outside_overlap_fraction': float(np.mean((p < lo) | (p > hi))),
            'clipped_fraction': float(clipped.mean()),
            'ess_control': ess_ctrl,
            'ess_treated': ess_case,
            'ess_control_fraction': ess_ctrl / max(len(p_ctrl), 1),
            'ess_treated_fraction': ess_case / max(len(p_case), 1),
            'score_q01': float(np.quantile(p, 0.01)),
            'score_median': float(np.median(p)),
            'score_q99': float(np.quantile(p, 0.99)),
        })
    return pd.DataFrame(rows)


def plot_propensity_scores(
    A, pi_hat, treatments=None, treatment_names=None, overlap_bounds=(0.05, 0.95),
    bins=40, max_panels=4, axes=None,
):
    """Plot propensity distributions for treatment and control cells."""
    A, pi_hat, treatment_names = _coerce_treatment_inputs(
        A, pi_hat, treatment_names)
    summary = summarize_propensity_scores(
        A, pi_hat, treatment_names=treatment_names,
        overlap_bounds=overlap_bounds, bins=bins,
    )

    if treatments is None:
        indices = list(range(min(A.shape[1], max_panels)))
    else:
        indices = []
        for treatment in treatments:
            if treatment in treatment_names:
                indices.append(treatment_names.index(treatment))
            else:
                index = int(treatment)
                if index < 0 or index >= A.shape[1]:
                    raise ValueError(f'Unknown treatment: {treatment}')
                indices.append(index)
    if not indices:
        raise ValueError('At least one treatment must be selected')

    if axes is None:
        ncols = min(2, len(indices))
        nrows = math.ceil(len(indices) / ncols)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False,
        )
        axes_flat = axes.ravel()
    else:
        axes_flat = np.asarray(axes, dtype=object).ravel()
        if len(axes_flat) < len(indices):
            raise ValueError('Not enough axes for the selected treatments')
        fig = axes_flat[0].figure

    ctrl = np.sum(A, axis=1) == 0
    for ax, j in zip(axes_flat, indices):
        case = A[:, j] == 1
        ax.hist(
            pi_hat[ctrl, j], bins=bins, range=(0, 1), density=True,
            histtype='step', linewidth=1.6, label='control', color='#4c78a8',
        )
        ax.hist(
            pi_hat[case, j], bins=bins, range=(0, 1), density=True,
            histtype='step', linewidth=1.6, label=str(treatment_names[j]),
            color='#e45756',
        )
        ax.axvline(overlap_bounds[0], color='#777777', linestyle='--', linewidth=0.8)
        ax.axvline(overlap_bounds[1], color='#777777', linestyle='--', linewidth=0.8)
        overlap = summary.loc[j, 'overlap_ratio']
        ess = summary.loc[j, 'ess_treated_fraction']
        ax.set_title(f'{treatment_names[j]} | overlap={overlap:.2f}, ESS={ess:.2f}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Estimated propensity score')
        ax.set_ylabel('Density')
        ax.legend(frameon=False)
    for ax in axes_flat[len(indices):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig, axes, summary
