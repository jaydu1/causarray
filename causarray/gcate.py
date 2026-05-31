from causarray.gcate_opt import *
import contextlib
import pandas as pd
from causarray.utils import comp_size_factor, _filter_params
import causarray.gcate_glm as _gcate_glm  # module-qualified so _USE_FAST_BACKEND changes take effect at call time


def _check_input(Y, X, family, disp_glm, disp_family, offset, c1, **kwargs):
    if not (X.ndim == 2 and Y.ndim == 2):
        raise ValueError("Input must have ndim of 2. Y.ndim: {}, X.ndim: {}.".format(Y.ndim, X.ndim))

    if not np.allclose(Y, Y.astype(int)):
        warnings.warn("Y is not integer-valued. It will be rounded to the nearest integer.")
        Y = np.round(Y)

    if np.sum(np.any(Y!=0., axis=0))<Y.shape[1]:
        raise ValueError("Y contains non-expressed features.")

    if np.linalg.svd(X, compute_uv=False)[-1] < 1e-3:
        raise ValueError("The covariate matrix is near singular.")
    
    Y = np.asarray(Y).astype(type_f)
    n, p = Y.shape

    kwargs_glm = {}
    kwargs_glm['family'] = family

    if offset is not None:
        if type(offset)==bool and offset is True:
            size_factor = comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs))
            kwargs_glm['size_factor'] = size_factor
            offset = np.log(size_factor)
        else:
            offset = np.asarray(offset)
    else:
        offset = None
    if kwargs_glm['family']=='nb':
        if disp_family is None:
            disp_family = 'poisson'
        if disp_glm is None:
            disp_glm = _gcate_glm.estimate_disp_auto(Y, X, offset=offset, disp_family=disp_family, maxiter=1000, **kwargs)
    if disp_glm is not None:
        kwargs_glm['disp_glm'] = disp_glm
            
    kwargs_glm = {**{'family':'gaussian', 'disp_glm':np.ones((1,p)), 'size_factor':np.ones((n,1))
    }, **kwargs_glm}

    c1 = 0.05 if c1 is None else c1
    lam1 = c1 #* a #* np.sqrt(np.log(p)/n)

    return Y, kwargs_glm, lam1


def fit_gcate(Y, X, A, r, family='nb', disp_glm=None, disp_family=None, offset=True,
    kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    c1=None, backend: str = "auto", A_init=None, **kwargs
):
    '''Fit the GCATE model.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        The response variable.
    X : array-like, shape (n, d)
        The covariate matrix.
    A : array-like, shape (n, a)
        The treatment matrix.
    r : int
        The number of unmeasured confounders.
    family : str
        The family of the GLM. Default is \'poisson\'.
    disp_glm : array-like, shape (p, ) or None
        The dispersion parameter for the negative binomial distribution.
    offset : array-like, shape (p, ) or None
        The offset parameter.
    kwargs_ls_1 : dict
        Keyword arguments for the line search solver in the first phrase.
    kwargs_ls_2 : dict
        Keyword arguments for the line search solver in the second phrase.
    kwargs_es_1 : dict
        Keyword arguments for the early stopper in the first phrase.
    kwargs_es_2 : dict
        Keyword arguments for the early stopper in the second phrase.
    c1 : float
        The regularization constant in the first phrase. Default is 0.1.
    backend : str
        GLM backend to use: ``"auto"`` (default), ``"fast"`` (force crispyx),
        or ``"original"`` (force statsmodels).  Not thread-safe.
    A_init : array-like, shape (n, d + a + r) or None
        Optional warm-start for the first-stage augmented factor matrix
        ``[X | A | U]`` passed to :func:`alter_min`.  When provided, the
        SVD-based initialisation in :func:`alter_min` is skipped and the
        supplied matrix seeds the first-stage iterations.  Used by
        :func:`fit_gcate_batch` to warm-start ``U`` rows for control cells
        across batches.
    kwargs : dict
        Additional keyword arguments.
    '''

    X = np.hstack((X, A))
    a = A.shape[1]
    ctx = _gcate_glm._backend_override(backend) if backend != "auto" else contextlib.nullcontext()
    with ctx:
        Y, kwargs_glm, lam1 = _check_input(Y, X, family, disp_glm, disp_family, offset, c1, **kwargs)

        r = int(r)

        res_1, res_2 = estimate(Y, X, r, a,
            lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2,
            A_init=A_init, **kwargs)

    return res_1, res_2


def estimate(Y, X, r, a, lam1,
    kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2,
    A_init=None, **kwargs):
    '''
    Two-stage estimation of the GCATE model.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    X : array-like, shape (n, d+a)
        Observed covariate matrix.
    r : int
        Number of latent variables.
    a : int
        The number of columns to be regularized. Assume the last 'a' columns of the covariates are the regularized coefficients. If 'a' is None, it is set to be 'd' by default.
    lam1 : float
        Regularization parameter for the first optimization problem.
    kwargs_glm : dict
        Keyword arguments for the GLM.
    kwargs_ls_1 : dict
        Keyword arguments of the line search algorithm for the first optimization problem.
    kwargs_ls_2 : dict
        Keyword arguments of the line search algorithm for the second optimization problem.
    kwargs_es_1 : dict
        Keyword arguments of the early stopping monitor for the first optimization problem.
    kwargs_es_2 : dict
        Keyword arguments of the early stopping monitor for the second optimization problem.
    kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    res_1 : dict
        The results of the first optimization problem.
    res_2 : dict
        The results of the second optimization problem, including 
            'X_U': the matrix (n, d+a+r) for the covariate and updated latent factors.
            'B_Gamma': the matrix (p, d+a+r) for the updated covariate and latent coefficients.
    '''

    p = Y.shape[1]

    valid_params = _filter_params(alter_min, kwargs)
    # ``A`` is controlled exclusively via the explicit ``A_init`` route below
    # so a stray ``A=`` kwarg from upstream cannot silently override it.
    valid_params.pop('A', None)

    res_1 = alter_min(
        Y, r, X=X, P1=True, A=A_init,
        kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_1, kwargs_es=kwargs_es_1, **valid_params)
    Q, _ = sp.linalg.qr(res_1['B_Gamma'][:,-r:], mode='economic')
    # Store thin Q factor (p, r) instead of the dense (p, p) projection
    # matrix.  For Adamson (p=14618, r=5) the shape change alone reclaims
    # ~1.7 GB — the (p, p) form was the actual cost, not the dtype.  No
    # ``.astype(np.float32)`` here because ``alter_min`` immediately
    # recasts ``P2`` to ``type_f`` (float64), so a float32 round-trip
    # would only allocate a transient extra ~0.3 MB without changing
    # peak memory.
    P_Gamma = Q

    if lam1 == 0.:
        res_2 = {'X_U': res_1['X_U'], 'B_Gamma': res_1['B_Gamma']}
    else:
        res_2 = alter_min(
            Y, r, X=X, P2=P_Gamma, A=res_1['X_U'].copy(), B=res_1['B_Gamma'].copy(), lam=lam1, a=a,
            kwargs_glm=res_1['kwargs_glm'], kwargs_ls=kwargs_ls_2, kwargs_es=kwargs_es_2, **valid_params)

    return res_1, res_2


def estimate_r(Y, X, A, r_max, c=1., 
    family='nb', disp_glm=None, disp_family='poisson', offset=True,
    max_cells=None, random_state=0,
    kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    **kwargs
):
    '''
    Estimate the number of latent factors for the GCATE model.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    X : array-like, shape (n, d)
        Observed covariate matrix.
    A : array-like, shape (n, a)
        Treatment matrix.
    r_max : int
        Number of latent variables.
    c : float
        The constant factor for the complexity term.
    family : str
        The family of the GLM. Default is 'poisson'.
    disp_glm : array-like, shape (1, p) or None
        The dispersion parameter for the negative binomial distribution.
    max_cells : int or None
        Maximum number of cells to use for estimation.  When the dataset
        exceeds this size, a stratified subsample is drawn automatically:
        a floor of ``max_cells // 4`` slots is reserved for treated cells
        (or all of them if ``len(pert_idx) < max_cells // 4``) so that
        perturbation-induced latent variation remains visible to the JIC,
        and the remaining budget goes to controls (with leftovers spilling
        back to treated cells if controls are themselves scarce).  This is
        especially useful for large batch-fitting workflows where ``n`` is
        in the tens of thousands; confounding structure (captured by ``r``)
        is concentrated in the baseline transcriptome, so a ctrl-priority
        subsample with a treated-cell floor is both faster and statistically
        principled.  ``None`` (default) uses all cells.
    random_state : int
        RNG seed for subsampling (only used when ``max_cells`` is set).
    kwargs_ls_1 : dict
        Keyword arguments of the line search algorithm for the first optimization problem.
    kwargs_ls_2 : dict
        Keyword arguments of the line search algorithm for the second optimization problem.
    kwargs_es_1 : dict
        Keyword arguments of the early stopping monitor for the first optimization problem.
    kwargs_es_2 : dict
        Keyword arguments of the early stopping monitor for the second optimization problem.

    Returns
    -------
    df_r : DataFrame
        Results of the number of latent factors.
    '''
    # ── Optional ctrl-priority subsampling ──────────────────────────────
    A_np = np.asarray(A) if not isinstance(A, pd.DataFrame) else A.values
    n_total = len(Y) if isinstance(Y, pd.DataFrame) else np.asarray(Y).shape[0]
    if max_cells is not None and n_total > max_cells:
        rng = np.random.default_rng(random_state)
        ctrl_mask = A_np.sum(axis=1) == 0
        ctrl_idx = np.where(ctrl_mask)[0]
        pert_idx  = np.where(~ctrl_mask)[0]
        # Budget split with a treated-cell floor.  Reserving at least
        # ``max_cells // 4`` slots for treated cells (or all of them if
        # ``len(pert_idx) < max_cells // 4``) ensures that perturbation-
        # induced latent variation is visible to the JIC even when the
        # control pool would otherwise saturate the budget — previously,
        # ``len(ctrl_idx) >= max_cells`` quietly left ``n_pert_use=0`` and
        # ``r`` was under-estimated.  After reserving the floor, any
        # remaining budget goes to controls; if controls are themselves
        # scarce, leftover slots fall back to additional treated cells.
        n_pert_floor = min(len(pert_idx), max(1, max_cells // 4))
        n_ctrl_use = min(len(ctrl_idx), max_cells - n_pert_floor)
        n_pert_use = min(len(pert_idx), max_cells - n_ctrl_use)
        sel_ctrl = rng.choice(ctrl_idx, n_ctrl_use, replace=False) if n_ctrl_use > 0 else np.array([], dtype=int)
        sel_pert = rng.choice(pert_idx, n_pert_use, replace=False) if n_pert_use > 0 else np.array([], dtype=int)
        sel = np.sort(np.concatenate([sel_ctrl, sel_pert]))
        Y = Y.iloc[sel] if isinstance(Y, pd.DataFrame) else np.asarray(Y)[sel]
        X = np.asarray(X)[sel]
        A = A.iloc[sel] if isinstance(A, pd.DataFrame) else A_np[sel]
        A_np = A_np[sel]
        if offset is not True and offset is not False and offset is not None:
            offset = np.asarray(offset)[sel]

    a, d = A_np.shape[1], np.asarray(X).shape[1]
    X = np.hstack((np.asarray(X), A_np))
    n, p = (len(Y), Y.shape[1]) if isinstance(Y, pd.DataFrame) else (np.asarray(Y).shape[0], np.asarray(Y).shape[1])

    Y, kwargs_glm, _ = _check_input(Y, X, family, disp_glm, disp_family, offset, None, **kwargs)
    
    family, nuisance, size_factor = kwargs_glm['family'], kwargs_glm['disp_glm'], kwargs_glm['size_factor']
    nuisance = nuisance.reshape(1,-1)
    size_factor = size_factor.reshape(-1,1)

    res = []
    if np.isscalar(r_max):
        r_list = np.arange(1, int(r_max)+1)
    else:
        r_list = np.array(r_max, dtype=int)
    r_max = np.max(r_list)

    # Estimate the residual deviance
    res_glm = _gcate_glm.fit_glm_auto(Y, X, offset=np.log(size_factor[:,0]), family=family, disp_glm=nuisance[0], maxiter=100, verbose=False)
    u, s, vt = svds(res_glm[-1], k=r_max)
    if u.shape[1]<r_max:
        raise ValueError(f'The number of latent factors is larger than the rank of deviance residuals ({u.shape[1]}). Try to decrease the value of r.')
    Q, _ = sp.linalg.qr(X, mode='economic')
    # Implicit projection: (I - Q @ Q.T) @ u = u - Q @ (Q.T @ u).
    # Avoids materialising the dense (n, n) matrix — for Adamson scale
    # (n ~ 30 000) this saves ~7 GB even when ``max_cells`` is not set.
    Q = Q.astype(type_f)
    u_proj = u - Q @ (Q.T @ u)
    A1 = np.c_[X, u_proj]

    logh = log_h(Y, family, nuisance)
    ll = 2 * ( 
        nll(Y, X, res_glm[0], family, nuisance, size_factor) / p 
        - np.sum(logh) / (n*p) ) 
    nu = (d+a) * np.maximum(n,p) * np.log(n * p / np.maximum(n,p)) / (n*p)
    jic = ll + c * nu
    res.append([0, ll, nu, jic])

    for r in r_list[r_list > 0][::-1]:
        _, res_2 = estimate(Y, X, r, a,
            0, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, A=A1[:,:d+a+r], **kwargs)
        A1, A2 = res_2['X_U'], res_2['B_Gamma']

        ll = 2 * ( 
            nll(Y, A1, A2, family, nuisance, size_factor) / p 
            - np.sum(logh) / (n*p) ) 
        nu = (d + a + r) * np.maximum(n,p) * np.log(n * p / np.maximum(n,p)) / (n*p)
        jic = ll + c * nu
        res.append([r, ll, nu, jic])

    df_r = pd.DataFrame(res, columns=['r', 'deviance', 'nu', 'JIC']).sort_values(by='r')
    return df_r 


def fit_gcate_batch(
    Y, X, A, r,
    batch_size=10,
    n_batches=None,
    max_cells=2000,
    n_ctrl=2000,
    family='nb',
    disp_glm=None,
    disp_family=None,
    offset=True,
    warm_start_U=False,
    skip_batches=None,
    random_state=0,
    verbose=False,
    **kwargs,
):
    """Fit GCATE independently on batches of perturbations.

    Partitions the ``a`` perturbations into chunks of ``batch_size`` and for
    each chunk selects the fixed ctrl subsample plus (a subset of) the treated
    cells, capped at ``max_cells`` total.  Dispersion is pre-estimated **once**
    on the ctrl cell pool so all batches share the same nuisance parameter
    (mirrors crispyx's Stage-1 global-model strategy).

    Parameters
    ----------
    Y : array-like, DataFrame, or scipy.sparse, shape (n, p)
        Count matrix.  Sparse inputs are densified to ``float64`` once at the
        start of the function; a :class:`ResourceWarning` is emitted when
        the dense materialisation would exceed ~4 GB.
    X : array, shape (n, d)
        Covariate matrix (intercept should be included).
    A : array-like or DataFrame, shape (n, a)
        Binary treatment indicator matrix; control cells have all-zero rows.
        **Single-perturbation-per-cell is assumed**: the batch loop treats
        any cell whose only active perturbation falls outside the current
        chunk as a control within that batch, which silently contaminates
        the within-batch null for combinatorial designs.  When such rows
        are detected a :class:`RuntimeWarning` is emitted at function
        entry — pass ``n_batches=1`` or pre-filter multi-pert rows to
        suppress it.
    r : int
        Number of latent factors.
    batch_size : int
        Perturbations per batch (default 10).  Ignored when ``n_batches`` is
        set.  Batches are sized evenly using :func:`numpy.array_split`, so
        the last batch is never more than one perturbation smaller than the
        others.
    n_batches : int or None
        Total number of batches.  When set, overrides ``batch_size`` and the
        perturbations are split as evenly as possible across exactly
        ``n_batches`` batches.  Useful when you want to control the number of
        batches rather than the per-batch perturbation count (e.g.
        ``n_batches=2`` splits a 29-pert dataset into two batches of 15/14).
    max_cells : int or None
        Maximum **pert** cells per batch (default 2 000).  ``None`` means no
        cap — all pert cells are used.  The ctrl pool is added on top, so the
        actual batch size is at most ``n_ctrl + max_cells``.  2 000 is a safe
        default because most Perturb-seq datasets have only a few hundred
        cells per perturbation, so the cap is rarely active.
    n_ctrl : int
        Number of ctrl cells in the fixed subsample shared across batches
        (default 2 000).
    family : str
        GLM family, ``'nb'`` or ``'poisson'`` (default ``'nb'``).
    disp_glm : array or None
        Dispersion parameter ``(p,)``.  If ``None`` and ``family='nb'``,
        estimated once on the ctrl cell subsample before the batch loop.
    disp_family : str or None
        Passed to ``fit_gcate`` (used only when ``disp_glm`` is ``None`` and
        the internal estimation path is taken inside each batch).
    offset : bool or array-like
        Offset specification passed to ``fit_gcate``.
    warm_start_U : bool
        If ``True``, initialises U rows for ctrl cells in batch ``i+1`` from
        the latent factors estimated in batch ``i``.  Requires the same ctrl
        indices in every batch (guaranteed by design).
    skip_batches : set of int or None
        Batch indices (0-based) to skip entirely.  Used by
        :func:`gcate_lfc_batch` to avoid re-running GCATE for batches whose
        LFC results are already cached on disk.  Skipped batches still appear
        in the returned list with ``'skipped': True`` and ``res_1/res_2 = None``.
    random_state : int
        Base RNG seed; each batch uses ``random_state + batch_i`` for pert
        subsampling to avoid drawing the same cells repeatedly.
    verbose : bool
        Print per-batch timing and progress.
    **kwargs
        Forwarded to :func:`fit_gcate` (e.g. ``backend``, ``kwargs_es_1``).

    Returns
    -------
    batch_results : list of dict
        One dict per batch with keys:

        ``'batch_i'``
            Batch index (0-based).
        ``'pert_names'``
            List of perturbation names in this batch.
        ``'ctrl_idx'``
            Global indices of ctrl cells (same for all batches).
        ``'pert_idx'``
            Global indices of pert cells used in this batch.
        ``'cell_idx'``
            Sorted union of ctrl and pert indices.
        ``'res_1'``, ``'res_2'``
            GCATE optimisation results (dicts from :func:`fit_gcate`).
        ``'disp_glm'``
            Shared dispersion array ``(p,)`` used for this batch.
        ``'t_batch'``
            Wall-clock seconds for this batch.
        ``'skipped'``
            ``True`` when the batch was in ``skip_batches``; absent otherwise.
    """
    import gc
    import time

    import scipy.sparse as _sp
    from causarray.utils import subsample_ctrl_cells, subsample_pert_cells, comp_size_factor, _filter_params
    from causarray.nb_glm_fast import _maybe_densify

    # ── Normalise inputs ────────────────────────────────────────────────
    # Sparse Y is common for Perturb-seq counts; densify with the same
    # memory-cost warning as ``_maybe_densify`` rather than crash on the
    # 0-d wrapper that ``np.asarray(sparse_matrix)`` would return.
    if _sp.issparse(Y):
        Y_np = _maybe_densify(Y)
    elif isinstance(Y, pd.DataFrame):
        Y_np = Y.values
    else:
        Y_np = np.asarray(Y)
    X_np = np.asarray(X)

    if isinstance(A, pd.DataFrame):
        pert_names_all = list(A.columns)
        A_np = A.values.astype(float)
    else:
        A_np = np.asarray(A, dtype=float)
        pert_names_all = list(range(A_np.shape[1]))

    n, p = Y_np.shape
    a_total = A_np.shape[1]

    # ── Multi-perturbation contamination check ─────────────────────────
    # The batch loop builds ``A_b`` from the chunk's columns only.  A cell
    # whose perturbation is OUTSIDE the current chunk surfaces in ``A_b``
    # as an all-zero row, indistinguishable from a true control — so its
    # expression (which still reflects the off-chunk perturbation) leaks
    # into the within-batch null estimate.  This is fine for single-pert
    # Perturb-seq (the dominant Perturb-seq design), but for combinatorial
    # screens it silently biases the per-batch GCATE fit.  Warn once if
    # multi-pert rows are detected so the user can decide whether to
    # process them in a single batch (``n_batches=1``) or accept the
    # contamination.
    _multi_pert_rows = int(np.sum(A_np.sum(axis=1) > 1))
    if _multi_pert_rows > 0:
        import warnings as _warnings
        _warnings.warn(
            f"fit_gcate_batch detected {_multi_pert_rows} cell(s) with more "
            f"than one active perturbation. Within each batch, off-chunk "
            f"perturbations are treated as controls, which biases the "
            f"per-batch null for combinatorial designs.  For pure "
            f"combinatorial screens pass ``n_batches=1`` (no chunking) or "
            f"exclude multi-pert rows before calling.",
            RuntimeWarning, stacklevel=2,
        )

    # ── Fixed ctrl subsample (shared across all batches) ────────────────
    ctrl_idx_all = np.where(A_np.sum(axis=1) == 0)[0]
    ctrl_sel = subsample_ctrl_cells(ctrl_idx_all, n_ctrl=n_ctrl, random_state=random_state)

    # ── Pre-estimate dispersion on ctrl cells (D1) ──────────────────────
    if family == 'nb' and disp_glm is None:
        if verbose:
            import pprint as _pprint
            _pprint.pprint('Pre-estimating dispersion on ctrl cell subsample...')
        if offset is True:
            offset_ctrl = np.log(comp_size_factor(Y_np[ctrl_sel]))
        elif offset is not None and offset is not False:
            offset_ctrl = np.asarray(offset)[ctrl_sel]
        else:
            offset_ctrl = None
        disp_glm = _gcate_glm.estimate_disp_auto(
            Y_np[ctrl_sel], X_np[ctrl_sel], offset=offset_ctrl)

    # ── Batch loop ───────────────────────────────────────────────────────
    # Determine number of batches: n_batches overrides batch_size.
    # np.array_split distributes the remainder evenly so the last batch is
    # never drastically smaller than the others (avoids a single-pert tail
    # batch that destabilises the propensity model).
    import math
    if n_batches is None:
        n_batches = math.ceil(a_total / batch_size)
    n_batches = max(1, min(n_batches, a_total))  # clamp to [1, a_total]
    chunks = [list(c) for c in np.array_split(range(a_total), n_batches) if len(c) > 0]

    batch_results = []
    U_ctrl_prev = None  # for warm_start_U
    t_total = 0.0

    _skip = set(skip_batches) if skip_batches is not None else set()

    _tqdm = tqdm if verbose else (lambda x, **kw: x)
    for batch_i, chunk_cols in enumerate(_tqdm(chunks, desc='GCATE batches', unit='batch')):
        chunk_pert_names = [pert_names_all[j] for j in chunk_cols]

        # ── Fast path: batch already cached — emit placeholder and skip ──
        if batch_i in _skip:
            batch_results.append({
                'batch_i': batch_i,
                'pert_names': chunk_pert_names,
                'ctrl_idx': ctrl_sel,
                'pert_idx': None,
                'cell_idx': None,
                'res_1': None,
                'res_2': None,
                'disp_glm': disp_glm,
                't_batch': 0.0,
                'skipped': True,
            })
            continue

        t0 = time.perf_counter()

        # Cells belonging to any pert in this chunk
        pert_idx = np.where(A_np[:, chunk_cols].sum(axis=1) > 0)[0]
        if max_cells is None:
            pert_sel = pert_idx  # no cap: keep all pert cells
        else:
            pert_sel = subsample_pert_cells(
                pert_idx, max_cells=max_cells, random_state=random_state + batch_i)

        cell_idx = np.sort(np.concatenate([ctrl_sel, pert_sel]))

        Y_b = Y_np[cell_idx]
        X_b = X_np[cell_idx]
        A_b = A_np[cell_idx][:, chunk_cols]

        # Build kwargs for fit_gcate; disp_glm is already estimated
        gcate_kw = dict(kwargs)
        gcate_kw['disp_glm'] = disp_glm
        # Suppress internal re-estimation by setting disp_family to 'poisson'
        # (ignored when disp_glm is provided, but be explicit)
        if 'disp_family' not in gcate_kw:
            gcate_kw['disp_family'] = 'poisson' if disp_glm is not None else disp_family

        # Warm-start U for ctrl rows from previous batch.  The matrix is
        # passed via fit_gcate's explicit ``A_init`` parameter so it lands on
        # alter_min's top-level ``A`` argument; stuffing it into
        # ``kwargs_ls_1`` would silently no-op because alter_min never reads
        # ``kwargs_ls['A']``.
        if warm_start_U and U_ctrl_prev is not None:
            # Map ctrl_sel → local row indices in cell_idx
            ctrl_local = np.searchsorted(cell_idx, ctrl_sel)
            n_b = len(cell_idx)
            d_b = X_b.shape[1] + len(chunk_cols)  # d_cov + a_batch
            X_U_init = np.zeros((n_b, d_b + r), dtype=np.float32)
            X_U_init[:, :d_b] = np.c_[X_b, A_b]
            X_U_init[ctrl_local, d_b:] = U_ctrl_prev
            gcate_kw['A_init'] = X_U_init

        res_1, res_2 = fit_gcate(Y_b, X_b, A_b, r, family=family, offset=offset, **gcate_kw)

        if warm_start_U:
            d_b = X_b.shape[1] + len(chunk_cols)
            ctrl_local = np.searchsorted(cell_idx, ctrl_sel)
            U_ctrl_prev = res_2['X_U'][ctrl_local, d_b:].copy()

        t_batch = time.perf_counter() - t0
        t_total += t_batch

        if verbose:
            t_avg = t_total / (batch_i + 1)
            eta = t_avg * (n_batches - batch_i - 1)
            import pprint as _pprint
            _pprint.pprint(
                f'Batch {batch_i+1}/{n_batches}: {len(chunk_pert_names)} perts, '
                f'{len(cell_idx)} cells, {t_batch:.1f}s  '
                f'(avg {t_avg:.1f}s/batch, ETA {eta/3600:.1f}h)')

        batch_results.append({
            'batch_i': batch_i,
            'pert_names': chunk_pert_names,
            'ctrl_idx': ctrl_sel,
            'pert_idx': pert_sel,
            'cell_idx': cell_idx,
            'res_1': res_1,
            'res_2': res_2,
            'disp_glm': disp_glm,
            't_batch': t_batch,
        })

    return batch_results