"""Diagnostics for treatment overlap, propensity scores, and result masks."""

import math
from typing import Hashable, Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.multitest import multipletests


_TestMask = Union[np.ndarray, pd.DataFrame, pd.Series]


def _normalize_test_labels(values, name):
    raw_labels = pd.Index(values)
    if raw_labels.hasnans:
        raise ValueError(f'{name} names must not be missing')
    labels = pd.Index([str(value) for value in raw_labels], dtype=object)
    if labels.has_duplicates:
        duplicates = labels[labels.duplicated()].unique().tolist()
        raise ValueError(f'{name} names must be unique; duplicates: {duplicates}')
    return labels


def _require_boolean_mask(values):
    values = np.asarray(values)
    if values.dtype.kind != 'b':
        raise ValueError('test_mask must contain only Boolean values')
    return values


def align_test_mask(
    results: pd.DataFrame,
    test_mask: _TestMask,
    treatment_names: Optional[Sequence[Hashable]] = None,
    gene_names: Optional[Sequence[Hashable]] = None,
    treatment_col: str = 'trt',
    gene_col: str = 'gene_names',
) -> np.ndarray:
    """Align a treatment-by-gene diagnostic mask to an LFC result table.

    Alignment uses treatment and gene labels rather than row position. This
    makes it possible to annotate or subset existing causarray results with a
    post-hoc diagnostic rule without refitting effects or changing their
    standard errors and p-values. Labels are compared after conversion to
    strings, matching the labeling convention of batch LFC results.

    Parameters
    ----------
    results : DataFrame
        Result table containing one row per treatment-gene test.
    test_mask : ndarray, DataFrame, or Series
        Boolean diagnostic mask. A two-dimensional array requires
        ``treatment_names`` and ``gene_names``. A DataFrame uses its index as
        treatment names and columns as gene names unless names are supplied
        explicitly. A Series must have a two-level ``(treatment, gene)``
        MultiIndex.
    treatment_names, gene_names : sequence, optional
        Labels for the rows and columns of a two-dimensional mask.
    treatment_col, gene_col : str, optional
        Columns containing treatment and gene labels in ``results``.

    Returns
    -------
    aligned : ndarray of bool, shape (n_tests,)
        Mask values in the original row order of ``results``.

    Raises
    ------
    ValueError
        If labels are missing or duplicated, the mask is malformed or
        non-Boolean, or a result test is absent from the mask.

    Notes
    -----
    This function only aligns a diagnostic mask. It does not modify the
    multiple-testing family or recompute adjusted p-values.

    .. versionadded:: 0.0.9
    """
    if not isinstance(results, pd.DataFrame):
        raise ValueError('results must be a pandas DataFrame')
    missing_columns = {treatment_col, gene_col}.difference(results.columns)
    if missing_columns:
        raise ValueError(
            f'results is missing required columns: {sorted(missing_columns)}'
        )
    if results[[treatment_col, gene_col]].isna().any().any():
        raise ValueError('result treatment and gene labels must not be missing')

    result_keys = pd.MultiIndex.from_arrays([
        results[treatment_col].astype(str),
        results[gene_col].astype(str),
    ])
    if result_keys.has_duplicates:
        duplicates = result_keys[result_keys.duplicated()].unique().tolist()
        raise ValueError(
            'results contains duplicate treatment-gene tests: '
            f'{duplicates[:5]}'
        )

    if isinstance(test_mask, pd.Series):
        if treatment_names is not None or gene_names is not None:
            raise ValueError(
                'treatment_names and gene_names must be omitted for a Series mask'
            )
        if (
            not isinstance(test_mask.index, pd.MultiIndex)
            or test_mask.index.nlevels != 2
        ):
            raise ValueError(
                'a Series test_mask must have a two-level treatment-gene MultiIndex'
            )
        values = _require_boolean_mask(test_mask.to_numpy())
        raw_treatments = pd.Index(test_mask.index.get_level_values(0))
        raw_genes = pd.Index(test_mask.index.get_level_values(1))
        if raw_treatments.hasnans or raw_genes.hasnans:
            raise ValueError(
                'test_mask treatment and gene labels must not be missing'
            )
        treatment_index = pd.Index(
            [str(value) for value in raw_treatments], dtype=object,
        )
        gene_index = pd.Index(
            [str(value) for value in raw_genes], dtype=object,
        )
        mask_index = pd.MultiIndex.from_arrays([treatment_index, gene_index])
        if mask_index.has_duplicates:
            duplicates = mask_index[mask_index.duplicated()].unique().tolist()
            raise ValueError(
                'test_mask contains duplicate treatment-gene tests: '
                f'{duplicates[:5]}'
            )
        lookup = pd.Series(values, index=mask_index)
    else:
        if isinstance(test_mask, pd.DataFrame):
            values = _require_boolean_mask(test_mask.to_numpy())
            if treatment_names is None:
                treatment_names = list(test_mask.index)
            if gene_names is None:
                gene_names = list(test_mask.columns)
        else:
            values = _require_boolean_mask(test_mask)
        if values.ndim != 2:
            raise ValueError('test_mask must be two-dimensional')
        if treatment_names is None or gene_names is None:
            raise ValueError(
                'treatment_names and gene_names are required for an array mask'
            )
        treatments = _normalize_test_labels(treatment_names, 'treatment')
        genes = _normalize_test_labels(gene_names, 'gene')
        expected_shape = (len(treatments), len(genes))
        if values.shape != expected_shape:
            raise ValueError(
                f'test_mask has shape {values.shape}; expected {expected_shape}'
            )
        mask_index = pd.MultiIndex.from_product([treatments, genes])
        lookup = pd.Series(values.ravel(), index=mask_index)

    aligned = lookup.reindex(result_keys)
    if aligned.isna().any():
        missing = result_keys[aligned.isna().to_numpy()].unique().tolist()
        raise ValueError(
            'test_mask is missing treatment-gene tests required by results: '
            f'{missing[:5]}'
        )
    return aligned.to_numpy(dtype=bool)


def _coerce_named_matrix(values, names, name, generated_prefix):
    if isinstance(values, pd.DataFrame):
        if names is None:
            names = list(values.columns)
        values = values.to_numpy()
    else:
        values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(f'{name} must be one- or two-dimensional')
    if names is None:
        names = [f'{generated_prefix}{j + 1}' for j in range(values.shape[1])]
    names = list(names)
    if len(names) != values.shape[1]:
        raise ValueError(f'{name} names must match the number of columns')
    if len(set(names)) != len(names):
        raise ValueError(f'{name} names must be unique')
    return values, names


def summarize_treatment_associations(
    A, Z, treatment_names=None, covariate_names=None, covariate_types=None,
):
    """Summarize covariate association with each treatment.

    Each treatment is compared with shared all-zero controls; rows assigned to
    other treatments are excluded. Spearman p-values are adjusted together
    across all finite treatment-by-covariate tests using the Benjamini-Hochberg
    procedure. The returned statistics are diagnostics and do not automatically
    identify variables that should be removed from an adjustment set.

    Parameters
    ----------
    A : array-like, shape (n, a)
        Binary treatment indicator matrix. DataFrame column names are retained.
    Z : array-like, shape (n, q)
        Observed covariates and/or estimated latent factors to diagnose.
    treatment_names : sequence, optional
        Treatment labels. Inferred from a DataFrame or generated when omitted.
    covariate_names : sequence, optional
        Covariate labels. Inferred from a DataFrame or generated when omitted.
    covariate_types : sequence, optional
        Labels such as ``'observed'`` and ``'latent'``. Defaults to
        ``'observed'`` for every column.

    Returns
    -------
    summary : DataFrame
        Long-form table containing ``spearman_rho``, ``pvalue``, globally
        BH-adjusted ``padj``, and ``standardized_mean_difference`` for every
        treatment-by-covariate pair.
    """
    A, treatment_names = _coerce_named_matrix(
        A, treatment_names, 'treatment', 'treatment_',
    )
    Z, covariate_names = _coerce_named_matrix(
        Z, covariate_names, 'covariate', 'covariate_',
    )
    if A.shape[0] != Z.shape[0]:
        raise ValueError('A and Z must have the same number of rows')
    if not np.all(np.isin(A, (0, 1))):
        raise ValueError('A must contain only binary treatment indicators')
    try:
        Z = np.asarray(Z, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('Z must contain numeric covariates') from exc
    if not np.all(np.isfinite(Z)):
        raise ValueError('Z must contain only finite values')

    if covariate_types is None:
        covariate_types = ['observed'] * Z.shape[1]
    covariate_types = list(covariate_types)
    if len(covariate_types) != Z.shape[1]:
        raise ValueError('covariate_types must match the number of covariates')

    ctrl = np.sum(A, axis=1) == 0
    if not np.any(ctrl):
        raise ValueError('At least one all-zero control row is required')

    rows = []
    for j, treatment in enumerate(treatment_names):
        case = A[:, j] == 1
        if not np.any(case):
            raise ValueError(f'Treatment {treatment} has no treated rows')
        eligible = ctrl | case
        y = A[eligible, j]
        for k, covariate in enumerate(covariate_names):
            values = Z[eligible, k]
            is_constant = bool(np.ptp(values) == 0)
            if is_constant:
                rho, pvalue = np.nan, np.nan
            else:
                rho, pvalue = stats.spearmanr(values, y)

            control_values = Z[ctrl, k]
            treated_values = Z[case, k]
            if len(control_values) < 2 or len(treated_values) < 2:
                smd = np.nan
            else:
                pooled_sd = np.sqrt(
                    (np.var(control_values, ddof=1)
                     + np.var(treated_values, ddof=1)) / 2
                )
                if pooled_sd == 0:
                    smd = np.nan
                else:
                    smd = (
                        np.mean(treated_values) - np.mean(control_values)
                    ) / pooled_sd
            rows.append({
                'treatment': treatment,
                'covariate': covariate,
                'covariate_type': covariate_types[k],
                'n_control': int(np.sum(ctrl)),
                'n_treated': int(np.sum(case)),
                'spearman_rho': float(rho),
                'pvalue': float(pvalue),
                'padj': np.nan,
                'standardized_mean_difference': float(smd),
                'constant': is_constant,
            })

    summary = pd.DataFrame(rows)
    finite = np.isfinite(summary['pvalue'].to_numpy())
    if np.any(finite):
        summary.loc[finite, 'padj'] = multipletests(
            summary.loc[finite, 'pvalue'], method='fdr_bh',
        )[1]
    return summary


def plot_treatment_associations(
    summary, value='spearman_rho', treatments=None, covariates=None,
    alpha=None, ax=None,
):
    """Plot a heatmap of treatment-by-covariate associations.

    Parameters
    ----------
    summary : DataFrame
        Output from :func:`summarize_treatment_associations`.
    value : {'spearman_rho', 'standardized_mean_difference'}, optional
        Statistic displayed in the heatmap.
    treatments, covariates : sequence, optional
        Ordered subsets to display.
    alpha : float or None, optional
        If provided, mark cells whose globally adjusted p-value is at most
        ``alpha``. No significance threshold is applied by default.
    ax : matplotlib Axes, optional
        Axes on which to draw the heatmap.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The populated figure and axes.
    """
    if value not in ('spearman_rho', 'standardized_mean_difference'):
        raise ValueError(
            "value must be 'spearman_rho' or 'standardized_mean_difference'"
        )
    if not isinstance(summary, pd.DataFrame):
        raise ValueError('summary must be a pandas DataFrame')
    required = {'treatment', 'covariate', value, 'padj'}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f'summary is missing required columns: {sorted(missing)}')
    if alpha is not None and not 0 < alpha <= 1:
        raise ValueError('alpha must be None or lie in (0, 1]')

    all_treatments = list(dict.fromkeys(summary['treatment']))
    all_covariates = list(dict.fromkeys(summary['covariate']))
    treatments = all_treatments if treatments is None else list(treatments)
    covariates = all_covariates if covariates is None else list(covariates)
    unknown_treatments = set(treatments).difference(all_treatments)
    unknown_covariates = set(covariates).difference(all_covariates)
    if unknown_treatments:
        raise ValueError(f'Unknown treatments: {sorted(unknown_treatments)}')
    if unknown_covariates:
        raise ValueError(f'Unknown covariates: {sorted(unknown_covariates)}')
    if not treatments or not covariates:
        raise ValueError('At least one treatment and covariate must be selected')

    subset = summary[
        summary['treatment'].isin(treatments)
        & summary['covariate'].isin(covariates)
    ]
    if subset.duplicated(['treatment', 'covariate']).any():
        raise ValueError('summary contains duplicate treatment-covariate pairs')
    matrix = subset.pivot(
        index='treatment', columns='covariate', values=value,
    ).reindex(index=treatments, columns=covariates)
    adjusted = subset.pivot(
        index='treatment', columns='covariate', values='padj',
    ).reindex(index=treatments, columns=covariates)

    if ax is None:
        fig, ax = plt.subplots(
            figsize=(max(6, 0.7 * len(covariates)),
                     max(3, 0.3 * len(treatments))),
        )
    else:
        fig = ax.figure
    values = matrix.to_numpy(dtype=float)
    finite_values = np.abs(values[np.isfinite(values)])
    limit = float(np.max(finite_values)) if finite_values.size else 1.0
    if limit == 0:
        limit = 1.0
    image = ax.imshow(values, aspect='auto', cmap='coolwarm',
                      vmin=-limit, vmax=limit)
    ax.set_xticks(np.arange(len(covariates)))
    ax.set_xticklabels(covariates, rotation=45, ha='right')
    ax.set_yticks(np.arange(len(treatments)))
    ax.set_yticklabels(treatments)
    ax.set_xlabel('Covariate or latent factor')
    ax.set_ylabel('Treatment')
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(
        'Spearman correlation' if value == 'spearman_rho'
        else 'Standardized mean difference'
    )
    if alpha is not None:
        padj = adjusted.to_numpy(dtype=float)
        for row, col in np.argwhere(np.isfinite(padj) & (padj <= alpha)):
            ax.text(col, row, '*', ha='center', va='center', color='black')
    fig.tight_layout()
    return fig, ax


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
