import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn_ensemble_cv import reset_random_seeds, Ensemble, ECV
from causarray.gcate_glm import fit_glm
import causarray.gcate_glm as _gcate_glm  # module-qualified so _USE_FAST_BACKEND changes take effect at call time
from causarray.utils import *
from causarray.utils import _filter_params
from joblib import Parallel, delayed
from tqdm import tqdm
import pprint
import warnings

from sklearn.model_selection import KFold, ShuffleSplit

def _get_func_ps(ps_model, **kwargs):
    if ps_model=='random_forest_cv':
        params_ps = _filter_params(fit_rf, kwargs)
        func_ps = lambda X, Y, X_test:fit_rf_ind_ps(X, Y[:,None], X_test=X_test, **params_ps)[:,0]
    elif ps_model=='logistic':
        clf_ps = LogisticRegression
        kwargs = {**{'fit_intercept':False, 'C':1e0, 'class_weight':None, 'random_state':0}, **kwargs}
        params_ps = _filter_params(clf_ps().get_params(), kwargs)
        func_ps = lambda X, Y, X_test: clf_ps(**params_ps).fit(X, Y).predict_proba(X_test)[:,1]
    elif ps_model=='ensemble':
        kwargs = dict(kwargs)
        logistic_class_weight = kwargs.pop('class_weight', None)
        params_ps_rf = _filter_params(fit_rf, kwargs)
        clf_ps = LogisticRegression
        kwargs = {**{
            'fit_intercept': False, 'C': 1e0,
            'class_weight': logistic_class_weight, 'random_state': 0,
        }, **kwargs}
        params_ps_lr = _filter_params(clf_ps().get_params(), kwargs)
        params_ps = {'params_ps_rf':params_ps_rf, 'params_ps_lr':params_ps_lr}
        func_ps = lambda X, Y, X_test:(fit_rf_ind(X, Y[:,None], X_test=X_test, **params_ps_rf)[:,0] + clf_ps(**params_ps_lr).fit(X, Y).predict_proba(X_test)[:,1])/2
    else:
        raise ValueError('Invalid propensity score model.')

    return func_ps, params_ps


def estimate_propensity_scores(
    A, X_A, K=1, ps_model='logistic', mask=None, clip=None,
    random_state=0, verbose=False, class_weight='balanced', **kwargs,
):
    """Estimate per-treatment propensity scores.

    Each treatment is compared with the shared all-zero control group.  With
    ``K > 1``, every returned score is predicted by a model that did not train
    on that cell. Logistic models use ``class_weight='balanced'`` by default,
    matching :func:`LFC` and historical causarray fits. Pass
    ``class_weight=None`` for calibrated treatment probabilities.

    Parameters
    ----------
    A : array-like, shape (n,) or (n, a)
        Binary treatment indicators.  Rows containing only zeros are controls.
    X_A : array-like, shape (n, d_A)
        Covariates used by the propensity model, including an intercept column
        when ``fit_intercept=False``.
    K : int, optional
        Number of folds.  ``1`` fits and predicts on all eligible cells;
        values greater than one produce out-of-fold predictions.
    ps_model : {'logistic', 'random_forest_cv', 'ensemble'}, optional
        Propensity model.
    mask : array-like or None, shape (n,) or (n, a)
        Optional per-treatment eligibility mask for model fitting.
    clip : tuple(float, float) or None, optional
        Bounds applied after prediction.  ``None`` returns raw probabilities.
    random_state : int, optional
        Random seed used for fold construction and supported estimators.
    class_weight : str, dict or None, optional
        Class weighting for logistic propensity estimation. The default
        ``'balanced'`` matches :func:`LFC`; pass ``None`` for calibrated
        probabilities.

    Returns
    -------
    pi_hat : ndarray, shape (n, a)
        Estimated probabilities ``P(A_j=1 | X_A)``.
    """
    A = np.asarray(A)
    if A.ndim == 1:
        A = A[:, None]
    X_A = np.asarray(X_A)
    if A.ndim != 2 or X_A.ndim != 2 or A.shape[0] != X_A.shape[0]:
        raise ValueError('A and X_A must be two-dimensional with matching rows')
    if not np.all(np.isin(A, (0, 1))):
        raise ValueError('A must contain only binary treatment indicators')

    try:
        K_int = int(K)
    except (TypeError, ValueError) as exc:
        raise ValueError('K must be a positive integer') from exc
    if K_int != K:
        raise ValueError('K must be a positive integer')
    K = K_int
    if K < 1:
        raise ValueError('K must be a positive integer')
    if K > A.shape[0]:
        raise ValueError('K cannot exceed the number of samples')

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim == 1:
            mask = mask[:, None]
        if mask.shape != A.shape:
            raise ValueError('Mask must have the same shape as the treatment matrix')

    func_ps, params_ps = _get_func_ps(
        ps_model, verbose=False, random_state=random_state,
        class_weight=class_weight, **kwargs)
    if verbose:
        pprint.pprint(params_ps)

    if ps_model == 'random_forest_cv':
        info_ecv = run_ecv(X_A, A, **params_ps)
        func_ps, params_ps = _get_func_ps(
            ps_model, verbose=False, ecv=False,
            kwargs_ensemble=info_ecv['best_params_ensemble'],
            kwargs_regr=info_ecv['best_params_regr'],
        )
        if verbose:
            pprint.pprint('Best parameters for the regression model:')
            pprint.pprint(info_ecv['best_params_regr'])
            pprint.pprint('Best parameters for the ensemble model:')
            pprint.pprint(info_ecv['best_params_ensemble'])

    n = A.shape[0]
    if K == 1:
        folds = [(np.arange(n), np.arange(n))]
    elif K == n:
        folds = [
            (np.delete(np.arange(n), j), np.array([j])) for j in range(n)
        ]
    else:
        folds = KFold(
            n_splits=K, random_state=random_state, shuffle=True,
        ).split(X_A)

    pi_hat = np.zeros(A.shape, dtype=float)
    for train_index, test_index in folds:
        A_train = A[train_index]
        XA_train, XA_test = X_A[train_index], X_A[test_index]
        i_ctrl = np.sum(A_train, axis=1) == 0

        for j in range(A.shape[1]):
            i_case = A_train[:, j] == 1
            eligible = mask[train_index, j] if mask is not None else (i_ctrl | i_case)
            y_train = A_train[eligible, j]
            if y_train.size == 0 or np.unique(y_train).size != 2:
                raise ValueError(
                    f'Treatment {j} needs at least one eligible control and case '
                    'in every training fold'
                )

            x_train = XA_train[eligible]
            constant_design = np.all(np.ptp(x_train, axis=0) == 0)
            if ps_model == 'logistic' and constant_design:
                if class_weight == 'balanced':
                    pi_hat[test_index, j] = 0.5
                elif isinstance(class_weight, dict):
                    sample_weight = np.asarray([
                        class_weight.get(int(value), 1.0) for value in y_train
                    ])
                    pi_hat[test_index, j] = np.average(
                        y_train, weights=sample_weight)
                else:
                    pi_hat[test_index, j] = np.mean(y_train)
            else:
                pi_hat[test_index, j] = func_ps(x_train, y_train, XA_test)

    if clip is not None:
        if len(clip) != 2 or not 0 <= clip[0] < clip[1] <= 1:
            raise ValueError('clip must be None or a pair 0 <= lower < upper <= 1')
        pi_hat = np.clip(pi_hat, clip[0], clip[1])
    return pi_hat


def cross_fitting(
    Y, A, X, X_A, family='poisson', K=1, glm_alpha=1e-4,
    ps_model='logistic', ps_class_weight='balanced',
    Y_hat=None, pi_hat=None, mask=None, ps_clip=(0.01, 0.99),
    return_raw_pi=False, verbose=False, **kwargs):
    '''
    Cross-fitting for causal estimands.

    Parameters
    ----------
    Y : array
        Outcomes.    
    A : array
        Binary treatment indicator.
    X : array
        Covariates.
    X_A : array
        Covariates for the propensity score model.
    family : str, optional
        The family of the generalized linear model. The default is 'poisson'.
    K : int, optional
        The number of folds for cross-validation. The default is 1.
    glm_alpha : float, optional
        The regularization parameter for the generalized linear model. The default is 1e-4.
    ps_model : str, optional
        The propensity score model. The default is 'logistic'.
    ps_class_weight : str, dict or None, optional
        Class weighting used by the propensity model. ``'balanced'`` preserves
        the established ``LFC`` nuisance fit; pass ``None`` for calibrated
        treatment probabilities.
    
    Y_hat : array, optional
        Estimated potential outcome of shape (n, p, a, 2). The default is None.
    pi_hat : array, optional
        Propensity score of shape (n, a). The default is None.
    mask : array, optional
        Boolean mask of shape (n, a) for the treatment, indicating which samples are used for 
        propensity-model fitting and the downstream estimand.
    ps_clip : tuple(float, float) or None, optional
        Bounds applied to scores used by AIPW. ``None`` disables clipping.
    return_raw_pi : bool, optional
        Return raw scores as a third result when true.

    **kwargs : dict
        Additional arguments to pass to the model.

    Returns
    -------    
    Y_hat : array
        Estimated potential outcome under control.
    pi_hat : array
        Estimated propensity score.
    pi_hat_raw : array
        Unclipped propensity score, returned only when ``return_raw_pi=True``.
    '''
    kwargs = dict(kwargs)
    if 'class_weight' in kwargs:
        legacy_class_weight = kwargs.pop('class_weight')
        warnings.warn(
            'Passing class_weight through LFC/cross_fitting is deprecated; '
            'use ps_class_weight instead.',
            FutureWarning, stacklevel=2,
        )
        ps_class_weight = legacy_class_weight

    params_glm = _filter_params(fit_glm, {**kwargs, 'verbose': verbose})

    if verbose:
        pprint.pprint(params_glm)
    
    if K > 1:
        n_samples = X.shape[0]
        if K >= n_samples:
            # Use Leave-One-Out Cross-Validation
            folds = [([i for i in range(n_samples) if i != j], [j]) for j in range(n_samples)]
        else:
            # Initialize KFold cross-validator
            kf = KFold(n_splits=int(K), random_state=0, shuffle=True)
            folds = kf.split(X)
    else:
        folds = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]

    # Initialize lists to store results
    if pi_hat is None:
        if verbose:
            pprint.pprint('Fit propensity score models...')
        ps_kwargs = {k: v for k, v in kwargs.items() if k != 'random_state'}
        if ps_model in ('logistic', 'ensemble'):
            ps_kwargs['class_weight'] = ps_class_weight
        pi_hat_raw = estimate_propensity_scores(
            A, X_A, K=K, ps_model=ps_model, mask=mask,
            random_state=kwargs.get('random_state', 0), verbose=verbose,
            **ps_kwargs,
        )
    else:
        pi_hat_raw = np.asarray(pi_hat, dtype=float).reshape(A.shape)
    if ps_clip is None:
        pi_hat = pi_hat_raw.copy()
    else:
        if len(ps_clip) != 2 or not 0 <= ps_clip[0] < ps_clip[1] <= 1:
            raise ValueError(
                'ps_clip must be None or a pair 0 <= lower < upper <= 1')
        pi_hat = np.clip(pi_hat_raw, ps_clip[0], ps_clip[1])
    fit_Y = True if Y_hat is None else False
    if fit_Y:
        _yhat_gb = Y.shape[0] * Y.shape[1] * A.shape[1] * 2 * 8 / 1e9
        _mem_limit_gb = kwargs.get('mem_limit_gb', None)
        if _mem_limit_gb is not None and _yhat_gb > _mem_limit_gb:
            warnings.warn(
                f"Y_hat allocation ({_yhat_gb:.1f} GB as float64) exceeds "
                f"mem_limit_gb={_mem_limit_gb} GB; using float32 to halve peak memory.",
                ResourceWarning, stacklevel=3,
            )
            Y_hat = np.zeros((Y.shape[0], Y.shape[1], A.shape[1], 2), dtype=np.float32)
        else:
            Y_hat = np.zeros((Y.shape[0], Y.shape[1], A.shape[1], 2), dtype=float)

    # Perform cross-fitting
    for train_index, test_index in folds:
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        XA_train, XA_test = X_A[train_index], X_A[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if fit_Y:
            if verbose: pprint.pprint('Fit outcome models...')
            # Subset offset to training fold (for fitting) and test fold (for
            # imputation) when it is a pre-computed array, so that
            # ``fit_glm_auto`` receives arrays with matching leading
            # dimensions in both stages.
            params_glm_fold = params_glm
            offset_test_arr = None
            if 'offset' in params_glm and isinstance(params_glm['offset'], np.ndarray):
                params_glm_fold = dict(params_glm)
                params_glm_fold['offset'] = params_glm['offset'][train_index]
                offset_test_arr = params_glm['offset'][test_index]
            # Fit GLM on training data and predict on test data
            res = _gcate_glm.fit_glm_auto(Y_train, X_train, A_train, family=family, alpha=glm_alpha,
                impute=X_test, offset_test=offset_test_arr, **params_glm_fold)
            Y_hat[test_index,:,:,0] = res[1][0]
            Y_hat[test_index,:,:,1] = res[1][1]

    Y_hat = np.clip(Y_hat, None, 1e5)
    if return_raw_pi:
        return Y_hat, pi_hat, pi_hat_raw
    return Y_hat, pi_hat





def AIPW_mean(Y, A, mu, pi):
    '''
    Augmented inverse probability weighted estimator (AIPW)

    Parameters
    ----------
    Y : array
        Outcomes of shape (n, p).
    A : array
        Binary treatment indicator of shape (n, a, 2).
    mu : array
        Conditional outcome distribution estimate of shape (n, p, a, 2).
    pi : array
        Propensity score of shape (n, a, 2).
    Returns
    -------
    tau : array
        A point estimate of the expected potential outcome of shape (p, a, 2).
    pseudo_y : array
        Pseudo-outcome of shape (n, p, a, 2).
    '''
    
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        weight = A / pi
    weight = weight[:, None, ...]
    Y = Y[:, :, None, None]

    # Influence-function values are intentionally left unconstrained.  Even
    # for a nonnegative outcome, individual AIPW pseudo-outcomes may be
    # negative; projecting them cell by cell changes their mean and biases the
    # estimator.  Parameter-space constraints belong after aggregation.
    pseudo_y = weight * (Y - mu) + mu

    tau = np.mean(pseudo_y, axis=0, dtype=np.float64)

    return tau, pseudo_y








def run_ecv(
    X, y, M=200, M_max=1000,
    # fixed parameters for bagging regressor
    kwargs_ensemble={},
    # fixed parameters for decision tree
    kwargs_regr={},
    # grid search parameters
    grid_regr={},
    grid_ensemble={}
):
    """
    Runs Ensemble Cross-Validation (ECV) to find the best hyperparameters.
    """
    kwargs_ensemble = {**{'verbose': 1, 'bootstrap': True}, **kwargs_ensemble}
    kwargs_regr = {**{'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'ccp_alpha': 0.02, 'class_weight': 'balanced'}, **kwargs_regr}
    grid_regr = {**{'max_depth': [3, 5, 7]}, **grid_regr}
    grid_ensemble = {**{'random_state': 0, 'max_samples': [0.4, 0.6, 0.8, 1.]}, **grid_ensemble}

    # Validate integer parameters
    M = int(M)
    M_max = int(M_max)

    # Make sure y is 2D
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    # Run ECV
    _, info_ecv = ECV(
        X, y, DecisionTreeClassifier, grid_regr, grid_ensemble,
        kwargs_regr, kwargs_ensemble,
        M=M, M0=M, M_max=M_max, return_df=True
    )

    # Replace the in-sample best parameter for 'n_estimators' with extrapolated best parameter
    info_ecv['best_params_ensemble']['n_estimators'] = info_ecv['best_n_estimators_extrapolate']

    return info_ecv


def fit_rf(
    X, y, X_test=None, M=100, M_max=1000, ecv=True,
    # fixed parameters for bagging regressor
    kwargs_ensemble={},
    # fixed parameters for decision tree
    kwargs_regr={},
    # grid search parameters
    grid_regr={},
    grid_ensemble={}
):
    """
    Fits a Random Forest model using parameters found by ECV.
    """

    kwargs_ensemble = {**{'verbose': 1, 'bootstrap': True}, **kwargs_ensemble}
    kwargs_regr = {**{'min_samples_split': 20, 'min_samples_leaf': 10, 'max_features': 'sqrt', 'ccp_alpha': 0.02, 'class_weight': 'balanced'}, **kwargs_regr}
    grid_regr = {**{'max_depth': [3, 5, 7]}, **grid_regr}
    grid_ensemble = {**{'random_state': 0, 'max_samples': [0.4, 0.6, 0.8, 1.]}, **grid_ensemble}

    # Make sure y is 2D
    y_2d = y.reshape(-1, 1) if y.ndim == 1 else y

    if ecv:
        # Get best parameters from ECV
        info_ecv = run_ecv(
            X, y_2d, M=M, M_max=M_max,
            kwargs_ensemble=kwargs_ensemble,
            kwargs_regr=kwargs_regr,
            grid_regr=grid_regr,
            grid_ensemble=grid_ensemble
        )
        params_regr = info_ecv['best_params_regr']
        params_ensemble = info_ecv['best_params_ensemble']
    else:
        params_regr = kwargs_regr
        params_ensemble = kwargs_ensemble
        
    # Fit the ensemble with the best CV parameters
    regr = Ensemble(
        estimator=DecisionTreeClassifier(**params_regr), **params_ensemble).fit(X, y_2d)

    # Predict
    if X_test is None:
        X_test = X
    return regr.predict(X_test).reshape(-1, y_2d.shape[1])



def fit_rf_ind(X, Y, *args, **kwargs):
    Y_hat = Parallel(n_jobs=-1)(delayed(fit_rf)(X, Y[:,j], *args, **kwargs)
        for j in tqdm(range(Y.shape[1])))
    Y_pred = np.concatenate(Y_hat, axis=-1)
    return Y_pred


def fit_rf_ind_ps(X, Y, *args, **kwargs):
    i_ctrl = (np.sum(Y, axis=1) == 0.)

    if 'X_test' not in kwargs:
        kwargs['X_test'] = X

    def _fit(X, y, i_ctrl, *args, **kwargs):        
        i_case = (y == 1.)
        i_cells = i_ctrl | i_case
        return fit_rf(X[i_cells], y[i_cells], *args, **kwargs)

    Y_hat = Parallel(n_jobs=-1)(delayed(_fit)(X, Y[:,j], i_ctrl, *args, **kwargs)
        for j in tqdm(range(Y.shape[1])))
    Y_pred = np.concatenate(Y_hat, axis=-1)

    return Y_pred


def fit_rf_ind_outcome(W, Y, A, *args, **kwargs):
    d = W.shape[1]
    a = A.shape[1]
    X = np.c_[W, A]
    X_test = np.tile(np.c_[W, np.zeros_like(A)][:,None,:], (1,1+a,1))
    for j in range(a):
        X_test[:,1+j,d+j] = 1
    X_test = X_test.reshape(-1, X_test.shape[-1])
    Y_pred = fit_rf_ind(X, Y, X_test=X_test)
    Y_pred = Y_pred.reshape(X.shape[0],1+a,Y.shape[1])
    Yhat_1 = Y_pred[:,1:,:].transpose(0,2,1)
    Yhat_0 = np.tile(Y_pred[:,0,:][:,:,None], (1,1,a))
    return Yhat_0, Yhat_1
