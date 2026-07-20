import numpy as np
import contextlib
import pandas as pd
import warnings
from typing import Literal
from causarray.DR_estimation import AIPW_mean, cross_fitting
from causarray.gcate_glm import loess_fit, ls_fit
import causarray.gcate_glm as _gcate_glm  # for _backend_override
from causarray.DR_inference import fdx_control, bh_correction
from causarray.utils import reset_random_seeds, pprint, tqdm, comp_size_factor, _filter_params


def _add_log2fc_columns(df_res):
    """Add base-2 LFC aliases while retaining natural-log result columns."""
    log2 = np.log(2.0)
    log2fc = df_res['tau'].to_numpy() / log2
    log2fc_se = df_res['std'].to_numpy() / log2

    # Recompute existing aliases as well as missing ones.  This normalizes
    # mixed old/new frames loaded from resumable batch caches and guarantees
    # that the aliases remain exact transformations of tau and std.
    for name in ('log2fc', 'log2fc_se'):
        if name in df_res.columns:
            df_res.pop(name)
    insert_at = df_res.columns.get_loc('std') + 1
    df_res.insert(insert_at, 'log2fc', log2fc)
    df_res.insert(insert_at + 1, 'log2fc_se', log2fc_se)
    return df_res



def compute_causal_estimand(
    estimand,
    Y, W, A, W_A=None, family='nb', offset=False,
    Y_hat=None, pi_hat=None, mask=None,
    fdx=False, fdx_B=1000, fdx_alpha=0.05, fdx_c=0.1,
    verbose=False, random_state=0, backend: str = "auto", K=1,
    ps_clip=(0.01, 0.99), ps_class_weight='balanced',
    **kwargs):
    """Estimate causal treatment effects using AIPW with a user-supplied estimand.

    Parameters
    ----------
    estimand : callable
        Function that maps influence function values ``(etas, A)`` to
        ``(eta_est, tau_est, var_est[, df_eff])``.  See :func:`LFC` for an
        example implementation.
    Y : array, shape (n, p)
        Count matrix of outcomes.
    W : array, shape (n, d)
        Covariate matrix (including latent factors from GCATE).
    A : array, shape (n, a)
        Binary treatment indicator matrix.
    W_A : array or None, shape (n, d_A)
        Covariate matrix for the propensity model.  If ``None``, ``W`` is used.
    family : str
        GLM family for the outcome model: ``'nb'`` (default) or ``'poisson'``.
    offset : bool or array-like
        Log-scale offset for the outcome model.  ``True`` computes size factors
        automatically; ``False`` or ``None`` disables the offset.
    Y_hat : array or None, shape (n, p, a, 2)
        Pre-computed counterfactual predictions.  When provided, cross-fitting
        is skipped.
    pi_hat : array or None, shape (n, a)
        Pre-computed propensity scores.  When provided, propensity fitting is
        skipped.
    K : int
        Number of folds used for nuisance estimation.  The default ``1``
        preserves in-sample fitting.
    ps_clip : tuple(float, float)
        Bounds applied to propensity scores used by AIPW.
    ps_class_weight : str, dict or None
        Class weighting for the propensity model. ``'balanced'`` preserves the
        established nuisance fit; pass ``None`` for calibrated probabilities.
    mask : array or None, shape (n, a)
        Boolean mask indicating eligible cells for each treatment. It limits
        propensity-model fitting and final estimand computation.
    fdx : bool
        Whether to apply FDX control (``P(FDP > fdx_c) < fdx_alpha``).
    fdx_B : int
        Number of bootstrap samples for FDX control.
    fdx_alpha : float
        Significance level for FDX control.
    fdx_c : float
        FDP threshold for FDX control.
    backend : str
        GLM backend: ``"auto"`` (default), ``"fast"`` (force crispyx),
        or ``"original"`` (force statsmodels).
    verbose : bool
        Print progress information.
    **kwargs
        Additional arguments forwarded to the GLM fitting functions.

    Returns
    -------
    df_res : DataFrame
        Test results produced by ``estimand``. An estimand may optionally return
        a fifth dictionary whose arrays are added as diagnostic columns.
    """
    reset_random_seeds(random_state)

    if 'clip_pseudo_outcomes' in kwargs:
        raise TypeError(
            'clip_pseudo_outcomes has been removed; AIPW pseudo-outcomes are '
            'always left unclipped'
        )

    ctx = _gcate_glm._backend_override(backend) if backend != "auto" else contextlib.nullcontext()

    # check the input data
    if isinstance(Y, pd.DataFrame):
        gene_names = Y.columns
        Y = Y.values
    else:
        gene_names = range(Y.shape[1])
    # When Y_hat (n×p×a×2 float64) would exceed mem_limit_gb, downcast to
    # float32 to halve peak memory.  Opt-in only: pass ``mem_limit_gb=<GB>``.
    _a_shape = A.shape[1] if hasattr(A, 'shape') and len(A.shape) > 1 else 1
    _yhat_gb = Y.shape[0] * Y.shape[1] * _a_shape * 2 * 8 / 1e9
    _mem_limit = kwargs.get('mem_limit_gb', None)
    _use_f32 = _mem_limit is not None and _yhat_gb > _mem_limit
    if _use_f32:
        warnings.warn(
            f"AIPW intermediate Y ({_yhat_gb:.1f} GB as float64) exceeds "
            f"mem_limit_gb={_mem_limit} GB; downcasting Y, A, and pi_hat to "
            f"float32 to halve peak memory.",
            ResourceWarning, stacklevel=2,
        )
    Y = Y.astype(np.float32 if _use_f32 else float)
    n, p = Y.shape
    if not np.all(np.isfinite(Y)):
        raise ValueError('Y must contain only finite values')

    if len(A.shape) == 1:
        A = A.reshape(-1,1)
    if isinstance(A, pd.DataFrame):
        trt_names = A.columns
        A = A.values
    else:
        trt_names = range(A.shape[1])

    if isinstance(W, pd.DataFrame):
        cov_names = W.columns
        W = W.values
    if W_A is None:
        W_A = W
    elif isinstance(W_A, pd.DataFrame):
        W_A = W_A.values

    if mask is not None:
        mask = np.array(mask).astype(bool)
        if len(mask.shape) == 1: mask = mask.reshape(-1,1)
        if mask.shape != A.shape:
            raise ValueError('Mask must have the same shape as the treatment matrix')

    kwargs = {k:v for k,v in kwargs.items() if k not in 
        ['kwargs_ls_1', 'kwargs_ls_2', 'kwargs_es_1', 'kwargs_es_2', 'c1', 'num_d']
    }

    if verbose:
        d_A = W_A.shape[1]
        pprint.pprint('Estimating LFC...')
        pprint.pprint({'estimands':'LFC','n':n,'p':p,'d':W.shape[1], 'd_A':d_A, 'a':A.shape[1]}, compact=True)

    if offset is not None and offset is not False:
        if type(offset)==bool and offset is True:
            size_factors = comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs))
            offset = np.log(size_factors)
        else:
            size_factors = np.exp(offset)
    else:
        offset = None
        size_factors = np.ones(n)
    size_factors = np.asarray(size_factors, dtype=float)
    if size_factors.shape != (n,) or not np.all(np.isfinite(size_factors)) or np.any(size_factors <= 0):
        raise ValueError('Size factors must be finite, positive, and have length n')
    
    with ctx:
        Y_hat, pi_hat, pi_hat_raw = cross_fitting(
            Y, A, W, W_A, family=family, K=K, offset=offset,
            Y_hat=Y_hat, pi_hat=pi_hat, mask=mask, ps_clip=ps_clip,
            ps_class_weight=ps_class_weight,
            return_raw_pi=True, random_state=random_state, verbose=verbose,
            **kwargs,
        )
    pi_hat = pi_hat.reshape(*A.shape)
    if not np.all(np.isfinite(Y_hat)):
        raise ValueError('Outcome-model predictions must contain only finite values')
    if not np.all(np.isfinite(pi_hat)) or np.any((pi_hat <= 0) | (pi_hat >= 1)):
        raise ValueError(
            'Propensity scores used by AIPW must be finite and strictly between 0 and 1; '
            'pass ps_clip bounds or provide valid pi_hat values')

    if verbose: pprint.pprint('Estimating AIPW mean...')
    _aipw_dtype = Y_hat.dtype if _use_f32 else None
    A_aipw     = A.astype(_aipw_dtype) if _aipw_dtype is not None else A
    pi_hat_aipw = pi_hat.astype(_aipw_dtype) if _aipw_dtype is not None else pi_hat

    # point estimation of the treatment effect
    _, etas = AIPW_mean(
        Y,
        np.stack([1-A_aipw, A_aipw], axis=-1),
        Y_hat,
        np.stack([1-pi_hat_aipw, pi_hat_aipw], axis=-1),
    )

    # normalize the influence function values
    etas /= size_factors[:,None,None,None]

    res = []
    iters = range(A.shape[1]) if A.shape[1]==1 else tqdm(range(A.shape[1]))
    for j in iters:
        if mask is not None:
            i_cells = mask[:, j]
        else:
            i_ctrl = (np.sum(A, axis=1) == 0.)
            i_case = (A[:,j] == 1.)
            i_cells = i_ctrl | i_case
        _ret = estimand(etas[i_cells,:,j], A[i_cells,j], **kwargs)
        eta_est, tau_est, var_est = _ret[:3]
        df_est = _ret[3] if len(_ret) > 3 else None
        estimand_info = _ret[4] if len(_ret) > 4 else None

        std_est = np.sqrt(var_est)
        tvalues_init = tau_est / std_est

        # Multiple testing procedure
        V = fdx_control(tau_est, tvalues_init, eta_est, std_est, fdx, fdx_B, fdx_alpha, fdx_c)

        # BH correction
        tvalues_init[np.isinf(std_est)] = np.nan
        pvals, qvals, pvals_adj, qvals_adj = bh_correction(tvalues_init, df=df_est)
        
        df_res = pd.DataFrame({
            'gene_names': gene_names,            
            'tau': tau_est,
            'std': std_est,
            'stat': tvalues_init,
            'rej': V,
            'pvalue': pvals,
            'padj': qvals,
            'pvalue_emp_null_adj': pvals_adj,
            'padj_emp_null_adj': qvals_adj,            
            })
        if estimand_info is not None:
            for name, values in estimand_info.items():
                df_res[name] = values
            n_nonestimable = int(np.sum(~np.asarray(estimand_info['estimable'], dtype=bool)))
            if n_nonestimable:
                treatment = trt_names[j] if A.shape[1] > 1 else 0
                warnings.warn(
                    f'{n_nonestimable} gene(s) for treatment {treatment!r} have '
                    'nonfinite or nonpositive AIPW counterfactual means and are '
                    'reported as non-estimable.',
                    RuntimeWarning, stacklevel=2,
                )
        if A.shape[1]>1:
            df_res['trt'] = trt_names[j]
        res.append(df_res)
    df_res = pd.concat(res, axis=0).reset_index(drop=True)
    estimation = {**{
        'pi_hat': pi_hat,
        'pi_hat_raw': pi_hat_raw,
        'Y_hat': Y_hat,
        'offset': offset,
        'size_factors': size_factors,
        'ps_class_weight': ps_class_weight,
    }, **kwargs}
    return df_res, estimation


def LFC(
    Y, W, A, W_A=None, family='nb', offset=False,
    Y_hat=None, pi_hat=None, cross_est=False, K=None, mask=None,
    usevar: Literal['unequal', 'pooled'] = 'unequal',
    thres_min=1e-2, thres_diff=1e-2, eps_var=1e-4,
    fdx=False, fdx_alpha=0.05, fdx_c=0.1,
    verbose=False, backend: str = "auto", ps_clip=(0.01, 0.99),
    ps_class_weight='balanced', **kwargs):
    """Estimate log-fold changes of treatment effects (LFCs) using AIPW.

    Fits a doubly-robust AIPW estimator for the log-ratio of counterfactual
    means E[Y(1)] / E[Y(0)].  Call this after :func:`fit_gcate` to incorporate
    estimated latent factors into the covariate matrix ``W``.

    Parameters
    ----------
    Y : array, shape (n, p)
        Count matrix of outcomes.
    W : array, shape (n, d)
        Covariate matrix, typically ``[X | U]`` where ``U`` are the latent
        factors from GCATE.
    A : array, shape (n, a)
        Binary treatment indicator matrix.
    W_A : array or None, shape (n, d_A)
        Covariate matrix for the propensity model.  If ``None``, ``W`` is used.
    family : str
        GLM family for the outcome model: ``'nb'`` (default) or ``'poisson'``.
    offset : bool or array-like
        Log-scale offset for the outcome model.  ``True`` computes size factors
        automatically; ``False`` or ``None`` disables the offset.
    Y_hat : array or None, shape (n, p, a, 2)
        Pre-computed counterfactual predictions.  When provided, cross-fitting
        is skipped.
    pi_hat : array or None, shape (n, a)
        Pre-computed propensity scores.  When provided, propensity fitting is
        skipped.
    cross_est : bool
        Whether to use two-fold cross-estimation for nuisance parameters.
        An explicit ``K`` takes precedence.
    K : int or None
        Number of nuisance-estimation folds.  ``None`` uses 2 when
        ``cross_est=True`` and 1 otherwise.
    mask : array or None, shape (n, a)
        Boolean mask indicating eligible cells for each treatment. It limits
        propensity-model fitting and final estimand computation.
    usevar : str
        Variance estimator for the AIPW pseudo-outcomes:

        * ``'unequal'`` (default, v0.0.6+): Welch variance
          ``s₀²/n₀ + s₁²/n₁`` with Welch-Satterthwaite degrees of
          freedom; p-values use the t-distribution. Prefer this estimator when
          treatment and control sample sizes or effective sample sizes are
          meaningfully unbalanced, when arm-specific pseudo-outcome variances
          may differ, and for case-control, bulk, and donor-level pseudo-bulk
          analyses. Independence of the rows does not imply equal
          treatment-arm variances, and biological heterogeneity or imbalance
          can make pooled inference anti-conservative.
        * ``'pooled'``: pooled-variance estimator ``(s² + eps_var) / n``.
          For a small, approximately balanced perturbation comparison, this
          estimator may provide better power when the independent sampling
          units and arm-specific pseudo-outcome variances are reasonably
          comparable. Treat it as an opt-in, empirically justified analysis,
          not as an automatic choice for every small study. Balanced sample
          counts alone are insufficient for case-control data, where biological
          heterogeneity commonly favors ``'unequal'``. Pooled inference can
          produce substantially smaller standard errors and many more
          discoveries.

        There is no universal arm-size ratio at which the choice should switch.
        Inspect nominal and propensity-weighted effective sample sizes,
        arm-specific pseudo-outcome variability, and sensitivity of the
        discoveries. When those diagnostics are uncertain, retain
        ``'unequal'``.

        ``'unequal'`` accommodates arm-specific variance but does not model
        within-donor or within-subject correlation. Repeated cells from the
        same biological unit should still be pseudo-bulked or analyzed with a
        cluster-aware method; changing ``usevar`` alone does not remove
        pseudoreplication.

        .. versionchanged:: 0.0.6
            Default changed from ``'pooled'`` to ``'unequal'``.  The
            ``'unequal'`` formula was also corrected from
            ``(s₀²/n₀ + s₁²/n₁)/2`` to the standard Welch form, which
            shrinks t-statistics by ≈ √2 relative to v0.0.5.  Pass
            ``usevar='pooled'`` to recover pre-v0.0.6 behaviour.
    thres_min : float
        Genes whose maximum counterfactual mean is below this threshold are
        excluded (reported as ``tau=0``, ``padj=NaN``).
    thres_diff : float
        Genes whose counterfactual means differ by less than this value are
        excluded.
    eps_var : float
        Small constant added to per-arm variances to prevent division by zero.
    fdx : bool
        Whether to apply FDX control (``P(FDP > fdx_c) < fdx_alpha``).
    fdx_alpha : float
        Significance level for FDX control.
    fdx_c : float
        FDP threshold for FDX control.
    verbose : bool
        Print progress information.
    backend : str
        GLM backend: ``"auto"`` (default), ``"fast"`` (force crispyx),
        or ``"original"`` (force statsmodels).
    ps_clip : tuple(float, float)
        Bounds applied to propensity scores used by AIPW. Raw, unclipped scores
        remain available as ``estimation['pi_hat_raw']``.
    ps_class_weight : str, dict or None
        Class weighting for the propensity model. ``'balanced'`` remains the
        default to limit nuisance-model drift; pass ``None`` for calibrated
        probabilities.
    **kwargs
        Additional arguments forwarded to the GLM fitting functions.

    Returns
    -------
    df_res : DataFrame
        Test results with natural-log effect estimate ``tau`` and standard
        error ``std``, together with their base-2 equivalents ``log2fc`` and
        ``log2fc_se``. The fold change is treatment relative to control, and
        ``log2fc_se`` is a standard error (not a sample standard deviation).
        The result also contains inference columns, raw ``mean_control`` and
        ``mean_treated`` counterfactual means, and an ``estimable`` flag (plus
        ``trt`` for multiple treatments).

        .. versionadded:: 0.0.8
            Added the ``log2fc`` and ``log2fc_se`` convenience columns. The
            original natural-log ``tau`` and ``std`` columns remain unchanged.
    """

    def estimand(etas, A, **kwargs):
        eta_0, eta_1 = etas[..., 0], etas[..., 1]
        mean_0 = np.mean(eta_0, axis=0, dtype=np.float64)
        mean_1 = np.mean(eta_1, axis=0, dtype=np.float64)
        finite_means = np.isfinite(mean_0) & np.isfinite(mean_1)
        estimable = finite_means & (mean_0 > 0) & (mean_1 > 0)

        # Apply the count-mean parameter-space constraint only after averaging
        # the unmodified AIPW pseudo-outcomes.  The floor makes the logarithm
        # and its delta-method denominator numerically safe.  Nonpositive raw
        # means remain non-estimable and are filtered below, so this safeguard
        # cannot turn an invalid aggregate into a discovery.
        tau_0 = np.where(estimable, np.maximum(mean_0, thres_diff), 1.0)
        tau_1 = np.where(estimable, np.maximum(mean_1, thres_diff), 1.0)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            tau_est = np.log(tau_1/tau_0)
            eta_est = eta_1 / tau_1[None,:] - eta_0 / tau_0[None,:]

        df_eff = None
        if usevar == 'pooled':
            var_est = (np.var(eta_est, axis=0, ddof=1) + eps_var) / eta_est.shape[0]
        elif usevar == 'unequal':
            # Welch variance: SE² = s₀²/n₀ + s₁²/n₁
            n_0 = int(np.sum(A==0))
            n_1 = int(np.sum(A==1))
            if n_0 < 2 or n_1 < 2:
                import warnings
                warnings.warn(
                    f"Welch variance requires at least 2 cells per arm; got "
                    f"n_0={n_0}, n_1={n_1} for this perturbation. Per-gene "
                    f"variance and df will be NaN; results for these genes "
                    f"are silently dropped by downstream BH correction. "
                    f"Pass ``usevar='pooled'`` if you need finite estimates "
                    f"in this regime.",
                    RuntimeWarning, stacklevel=3,
                )
            with np.errstate(invalid='ignore', divide='ignore'):
                var_0 = np.var(eta_est[A==0], axis=0, ddof=1)
                var_1 = np.var(eta_est[A==1], axis=0, ddof=1)
                v0 = (var_0 + eps_var) / n_0
                v1 = (var_1 + eps_var) / n_1
                var_est = v0 + v1
                # Welch-Satterthwaite degrees of freedom (per gene)
                df_eff = (v0 + v1)**2 / (v0**2 / (n_0 - 1) + v1**2 / (n_1 - 1))
        else:
            raise ValueError('usevar must be either "pooled" or "unequal"')

        # Filter on the raw aggregate estimates, not on cell-level projections
        # or the numerical floor used for the log transform.
        idx = (
            ~estimable |
            (np.maximum(mean_0, mean_1) < thres_min) |
            (np.abs(mean_1 - mean_0) < thres_diff)
        )
        tau_est[idx] = 0.; eta_est[:,idx] = 0.; var_est[idx] = np.inf
        if df_eff is not None:
            df_eff[idx] = np.nan

        # Count genes whose Welch df is NaN for reasons OTHER than the
        # low-expression filter (which is intentional) — typically caused by
        # ``var = NaN`` from very small per-arm counts.  These rows are
        # silently dropped by ``bh_correction``'s ``~np.isnan(t)`` mask, so
        # surface a single warning if any survive the filter.
        if df_eff is not None:
            silent_nan = int(np.sum(np.isnan(df_eff) & ~idx))
            if silent_nan > 0:
                import warnings
                warnings.warn(
                    f"{silent_nan} gene(s) produced NaN Welch df despite "
                    f"passing the low-expression filter; these will be "
                    f"reported as NaN p-values in the result DataFrame.",
                    RuntimeWarning, stacklevel=3,
                )

        info = {
            'mean_control': mean_0,
            'mean_treated': mean_1,
            'estimable': estimable,
        }
        return eta_est, tau_est, var_est, df_eff, info

    if K is None:
        K = 2 if cross_est else 1

    df_res, estimation = compute_causal_estimand(
        estimand, Y, W, A, W_A, family, offset,    
        Y_hat=Y_hat, pi_hat=pi_hat, mask=mask,
        fdx=fdx, fdx_alpha=fdx_alpha, fdx_c=fdx_c,
        verbose=verbose, backend=backend, K=K, ps_clip=ps_clip,
        ps_class_weight=ps_class_weight, **kwargs)
    return _add_log2fc_columns(df_res), estimation





def VIM(eta_est, X, id_covs, **kwargs):
    """Estimate variable importance measures (VIM) for heterogeneous treatment effects.

    Decomposes treatment effect variance into components explained by each
    covariate using conditional average treatment effect (CATE) regression.

    Parameters
    ----------
    eta_est : array, shape (n, p)
        Influence function values from :func:`LFC` or
        :func:`compute_causal_estimand`.
    X : array, shape (n, d)
        Covariate matrix.
    id_covs : int or array-like of int
        Column indices of ``X`` to compute VIM for.  An integer ``k`` is
        treated as ``range(k)``.

    Returns
    -------
    estimation : dict
        Dictionary with keys:

        ``'CATE'``, ``'CATE_lower'``, ``'CATE_upper'`` : array, shape (n_covs, n, p)
            Conditional average treatment effect and pointwise confidence band.
        ``'VTE'`` : array, shape (p,)
            Total variance of the treatment effect (marginal).
        ``'CVTE'`` : array, shape (n_covs, p)
            Conditional variance of the treatment effect given each covariate.
        ``'VIM_mean'`` : array, shape (n_covs, p)
            VIM point estimate ``CVTE / VTE - 1`` for each covariate and gene.
        ``'VIM_sd'`` : array, shape (n_covs, p)
            Standard deviation of the VIM estimate.
    """
    if len(X.shape)==1:
        X = X[:,None]

    n, p = eta_est.shape
    d = X.shape[1]
    if id_covs is None:
        id_covs = range(d)
    if np.isscalar(id_covs):
        id_covs = range(id_covs)

    n_covs = len(id_covs)

    emp_VTE = (eta_est - np.mean(eta_est, axis=0, keepdims=True))**2
    VTE = np.mean(emp_VTE, axis=0)
    VIM_mean = np.zeros((n_covs, p))
    VIM_sd = np.zeros((n_covs, p))
    emp_CVTE = np.zeros((n_covs, n, p))
    CVTE = np.zeros((n_covs, p))
    CATE = np.zeros((n_covs, n, p))
    CATE_lower = np.zeros((n_covs, n, p))
    CATE_upper = np.zeros((n_covs, n, p))

    for j,i in enumerate(id_covs):
        print(j,i)
        # regression eta_est on X to get predicted values
        if np.all(np.modf(X[:,i:i+1])[0] == 0):
            CATE[j], CATE_lower[j], CATE_upper[i] = ls_fit(eta_est, X[:,i], **kwargs)
        else:
            CATE[j], CATE_lower[j], CATE_upper[j] = loess_fit(eta_est, X[:,i], **kwargs)
        # compute the variance of treatment effect        
        _emp_CVTE = (eta_est - CATE[j])**2
        _CVTE = np.nanmean(_emp_CVTE, axis=0)
        emp_CVTE[j] = _emp_CVTE
        CVTE[j] = _CVTE

        VIM_mean[j] = _CVTE / VTE - 1
        VIM_sd[j] = np.nanstd((emp_VTE - _emp_CVTE), axis=0, ddof=1)/VTE

    estimation = {
        'CATE': CATE,
        'CATE_lower': CATE_lower,
        'CATE_upper': CATE_upper,
        'emp_VTE': emp_VTE,
        'VTE': VTE,
        'emp_CVTE' : emp_CVTE,
        'CVTE' :CVTE,
        'VIM_mean' : VIM_mean,
        'VIM_sd' : VIM_sd
    }
    return estimation


def gcate_lfc_batch(
    Y, X, A, r,
    W_A=None,
    batch_size=10,
    n_batches=None,
    max_cells=2000,
    n_ctrl=2000,
    family='nb',
    offset=True,
    warm_start_U=False,
    cache_path=None,
    random_state=0,
    verbose=False,
    gcate_kwargs=None,
    lfc_kwargs=None,
    **kwargs,
):
    """Batch-wise GCATE + doubly-robust LFC estimation.

    Partitions perturbations into chunks of ``batch_size``, runs
    :func:`fit_gcate_batch` to estimate per-batch latent confounders, then
    calls :func:`LFC` on each batch independently.  All large intermediate
    arrays (``res_1``, ``res_2``, ``Y_hat``, ``pi_hat``) are freed immediately
    after each batch so that peak memory is bounded by one batch's worth of
    data regardless of the total number of perturbations.

    Results can optionally be cached to an HDF5 file (``cache_path``) so that
    interrupted runs can be resumed without re-processing completed batches.

    Parameters
    ----------
    Y : array-like or DataFrame, shape (n, p)
        Count matrix.
    X : array, shape (n, d)
        Covariate matrix (intercept column should be included).
    A : array-like or DataFrame, shape (n, a)
        Binary treatment indicator matrix; control cells have all-zero rows.
    r : int
        Number of latent factors.
    W_A : array or None, shape (n, d_A)
        Propensity-score covariate matrix.  If ``None``, ``X`` is used.
    batch_size : int
        Perturbations per batch (default 10).  Ignored when ``n_batches`` is
        set.  Batches are sized evenly with :func:`numpy.array_split` so the
        last batch is never drastically smaller than the others.
    n_batches : int or None
        Total number of batches.  When set, overrides ``batch_size`` and
        perturbations are split as evenly as possible across exactly
        ``n_batches`` batches (e.g. ``n_batches=2`` on a 29-pert dataset
        gives two batches of 15 and 14).
    max_cells : int or None
        Maximum **pert** cells per batch (default 2 000).  ``None`` means no
        cap.  Ctrl cells are added on top so the actual batch size is at most
        ``n_ctrl + max_cells``.  The cap is rarely active because typical
        Perturb-seq datasets have only a few hundred cells per perturbation.
    n_ctrl : int
        Number of ctrl cells in the fixed subsample (default 2 000).
    family : str
        GLM family (default ``'nb'``).
    offset : bool or array-like
        Offset specification passed to :func:`fit_gcate_batch`.
    warm_start_U : bool
        Passed to :func:`fit_gcate_batch`.
    cache_path : str or None
        Path to an HDF5 file used for incremental caching.  When set:

        - On entry, any already-computed batches are loaded from the store and
          their indices are skipped by :func:`fit_gcate_batch`.
        - After each new batch, the result DataFrame is appended to the store
          under key ``/batch_{i:04d}``.
        - On exit, all batches (cached + newly computed) are concatenated and
          returned.

        This lets you resume an interrupted run by re-calling the function
        with the same ``cache_path`` — completed batches are not re-run.
    random_state : int
        RNG seed.
    verbose : bool
        Print per-batch timing.
    gcate_kwargs : dict or None
        Extra keyword arguments forwarded to :func:`fit_gcate_batch`
        (and ultimately :func:`fit_gcate`).  E.g.::

            gcate_kwargs=dict(backend='fast',
                              kwargs_es_1=dict(max_iters=10, rel_tol=2e-4),
                              kwargs_es_2=dict(max_iters=10, rel_tol=2e-4))

    lfc_kwargs : dict or None
        Extra keyword arguments forwarded to :func:`LFC`
        (e.g. ``usevar``, ``fdx``, ``thres_min``). Retain the default
        ``usevar='unequal'`` when arm sizes or effective sample sizes are
        meaningfully unbalanced, when arm-specific variability may differ, and
        for case-control, bulk, or pseudo-bulk analyses. For a small,
        approximately balanced perturbation comparison with comparable
        pseudo-outcome variability, ``lfc_kwargs=dict(usevar='pooled')`` may
        improve power. There is no universal balance threshold; compare the
        relevant diagnostics and retain ``'unequal'`` when uncertain.
    **kwargs
        Additional arguments forwarded to both :func:`fit_gcate_batch` and
        :func:`LFC`.  When a key collides with ``gcate_kwargs`` /
        ``lfc_kwargs``, the **stage-specific dict wins** — this lets you
        scope a kwarg to one stage (e.g. ``gcate_kwargs=dict(
        backend='fast')`` paired with a top-level ``backend='original'``
        targeting LFC).

    Returns
    -------
    df_res : DataFrame
        Concatenated result from all batches. Includes natural-log ``tau`` and
        ``std`` columns, base-2 ``log2fc`` and ``log2fc_se`` columns, and a
        ``'batch'`` column with the 0-based batch index. Older compatible
        caches containing only ``tau`` and ``std`` are upgraded in memory.
    """
    import gc
    from causarray.gcate import fit_gcate_batch

    if gcate_kwargs is None:
        gcate_kwargs = {}
    if lfc_kwargs is None:
        lfc_kwargs = {}

    import scipy.sparse as _sp
    from causarray.nb_glm_fast import _maybe_densify
    if _sp.issparse(Y):
        Y_np = _maybe_densify(Y)
        gene_names = None
    elif isinstance(Y, pd.DataFrame):
        Y_np = Y.values
        gene_names = list(Y.columns)
    else:
        Y_np = np.asarray(Y)
        gene_names = None  # will fall back to range(p) inside LFC
    X_np = np.asarray(X)
    W_A_np = np.asarray(W_A) if W_A is not None else None

    if isinstance(A, pd.DataFrame):
        A_np = A.values.astype(float)
        pert_names_all = list(A.columns)
    else:
        A_np = np.asarray(A, dtype=float)
        pert_names_all = list(range(A_np.shape[1]))

    # ── Disk-cache: identify cached batches and validate schema ─────────
    from causarray.__about__ import __version__ as _causarray_version
    cached_keys = {}  # batch_i → "/batch_NNNN" hdf5 key
    cache_meta_existing = None
    skip_batches = set()
    if cache_path is not None:
        try:
            with pd.HDFStore(cache_path, mode='r') as store:
                keys = store.keys()
                if '/meta' in keys:
                    cache_meta_existing = store['/meta']
                for key in keys:
                    # keys look like '/batch_0000'
                    if key.startswith('/batch_'):
                        try:
                            idx = int(key.split('_')[1])
                            cached_keys[idx] = key
                            skip_batches.add(idx)
                        except (ValueError, IndexError):
                            pass
        except (FileNotFoundError, OSError):
            pass  # cache file doesn't exist yet — start fresh
        if cache_meta_existing is not None:
            _expected = {
                'family': family,
                'a_total': int(A_np.shape[1]),
            }
            for key_, val_ in _expected.items():
                if key_ in cache_meta_existing.columns:
                    got = cache_meta_existing.iloc[0][key_]
                    if got != val_:
                        raise ValueError(
                            f"gcate_lfc_batch cache at {cache_path!r} was written "
                            f"with {key_}={got!r}, but the current call uses "
                            f"{key_}={val_!r}.  Refusing to mix incompatible "
                            f"schemas — delete the cache file or pass a fresh "
                            f"cache_path to start over."
                        )
        if verbose and skip_batches:
            print(f'[gcate_lfc_batch] Resuming: {len(skip_batches)} batches '
                  f'already cached in {cache_path!r}')

    # ── Run GCATE in batches ─────────────────────────────────────────────
    # Explicit ``gcate_kwargs`` wins over the generic ``**kwargs`` so a kwarg
    # can be scoped to one stage (e.g. ``gcate_kwargs=dict(backend='fast')``
    # with a top-level ``backend='original'`` targeting LFC only).
    batch_results = fit_gcate_batch(
        Y_np, X_np, A, r,
        batch_size=batch_size,        n_batches=n_batches,        max_cells=max_cells,
        n_ctrl=n_ctrl,
        family=family,
        offset=offset,
        warm_start_U=warm_start_U,
        skip_batches=skip_batches,
        random_state=random_state,
        verbose=verbose,
        **{**kwargs, **gcate_kwargs},
    )

    # Build a column-name lookup once
    if isinstance(A, pd.DataFrame):
        pert_col_map = {name: i for i, name in enumerate(pert_names_all)}
    else:
        pert_col_map = {i: i for i in range(A_np.shape[1])}

    new_dfs = {}
    for batch_i, br in enumerate(batch_results):
        # Skipped batches already have their DataFrame in cached_dfs
        if br.get('skipped'):
            continue

        cell_idx = br['cell_idx']
        chunk_pert_names = br['pert_names']
        res_2 = br['res_2']

        # Extract latent factors (copy needed — 'U' is a view of 'X_U')
        U_b = res_2['U'].copy()
        # Use offset already computed by fit_gcate (consistent with GCATE fitting)
        offset_b = np.log(res_2['kwargs_glm']['size_factor'])

        Y_b_np = Y_np[cell_idx]
        # Preserve gene names so df_b['gene_names'] has the same type as df_full
        Y_b = pd.DataFrame(Y_b_np, columns=gene_names) if gene_names is not None else Y_b_np
        X_b = X_np[cell_idx]

        # Recover pert columns for this batch
        chunk_cols = [pert_col_map[name] for name in chunk_pert_names]
        A_b = A_np[np.ix_(cell_idx, chunk_cols)]

        W_b = np.c_[X_b, U_b]
        if W_A_np is not None:
            W_A_b = np.c_[W_A_np[cell_idx], U_b]
        else:
            W_A_b = W_b

        A_df_b = pd.DataFrame(A_b, columns=chunk_pert_names)

        df_b, estimation_b = LFC(
            Y_b, W_b, A_df_b, W_A_b,
            family=family,
            offset=offset_b,
            verbose=verbose,
            **{**kwargs, **lfc_kwargs},
        )
        df_b['batch'] = batch_i
        new_dfs[batch_i] = df_b

        if cache_path is not None:
            with pd.HDFStore(cache_path, mode='a') as store:
                if '/meta' not in store.keys():
                    _meta = pd.DataFrame({
                        'causarray_version': [_causarray_version],
                        'family': [family],
                        'a_total': [int(A_np.shape[1])],
                        'columns': [','.join(df_b.columns.astype(str))],
                    })
                    store.put('/meta', _meta, format='fixed')
                store.put(f'batch_{batch_i:04d}', df_b, format='fixed')

        del estimation_b   # releases Y_hat and pi_hat
        del U_b, Y_b, Y_b_np, X_b, A_b, W_b, W_A_b
        br['res_1'] = None
        br['res_2'] = None
        gc.collect()

    new_indices = set(new_dfs)
    cached_indices = set(cached_keys)
    all_indices = sorted(new_indices | cached_indices)

    if cached_indices and cache_path is not None:
        with pd.HDFStore(cache_path, mode='r') as store:
            frames = [
                new_dfs[i] if i in new_indices else store[cached_keys[i]]
                for i in all_indices
            ]
            result = pd.concat(frames, axis=0).reset_index(drop=True)
    else:
        result = pd.concat(
            [new_dfs[i] for i in all_indices], axis=0
        ).reset_index(drop=True)
    return _add_log2fc_columns(result)


def LFC_batch(*args, **kwargs):
    """Deprecated alias for :func:`gcate_lfc_batch`.

    .. deprecated::
        Use ``gcate_lfc_batch`` instead.  ``LFC_batch`` will be removed in a
        future release.
    """
    import warnings
    warnings.warn(
        "LFC_batch is deprecated and will be removed in a future release. "
        "Use gcate_lfc_batch instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return gcate_lfc_batch(*args, **kwargs)
