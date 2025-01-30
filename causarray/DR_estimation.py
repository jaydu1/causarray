import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from causarray.gcate_glm import fit_glm
from causarray.utils import *
from causarray.utils import _filter_params
import pprint

from sklearn.model_selection import KFold, ShuffleSplit

def _get_func_ps(ps_model, **kwargs):
    if ps_model=='random_forest_cv':
        params_ps = _filter_params(fit_rf, kwargs)
        func_ps = lambda X, Y, X_test:fit_rf_ind_ps(X, Y[:,None], X_test=X_test, **params_ps)[:,0]
    elif ps_model=='logistic':
        clf_ps = LogisticRegression
        kwargs = {**{'fit_intercept':False, 'C':1e0, 'class_weight':'balanced', 'random_state':0}, **kwargs}
        params_ps = _filter_params(clf_ps().get_params(), kwargs)
        func_ps = lambda X, Y, X_test: clf_ps(**params_ps).fit(X, Y).predict_proba(X_test)[:,1]
    elif ps_model=='ensemble':
        params_ps_rf = _filter_params(fit_rf, kwargs)
        clf_ps = LogisticRegression
        kwargs = {**{'fit_intercept':False, 'C':1e0, 'class_weight':'balanced', 'random_state':0}, **kwargs}
        params_ps_lr = _filter_params(clf_ps().get_params(), kwargs)
        params_ps = {'params_ps_rf':params_ps_rf, 'params_ps_lr':params_ps_lr}
        func_ps = lambda X, Y, X_test:(fit_rf_ind(X, Y[:,None], X_test=X_test, **params_ps_rf)[:,0] + clf_ps(**params_ps_lr).fit(X, Y).predict_proba(X_test)[:,1])/2
    else:
        raise ValueError('Invalid propensity score model.')

    return func_ps, params_ps


def cross_fitting(
    Y, A, X, X_A, family='poisson', K=1, glm_alpha=1e-4,
    ps_model='logistic', 
    pi_hat=None, Y_hat=None, verbose=False, **kwargs):
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
    pi_hat : array, optional
        Propensity score of shape (n, a). The default is None.
    Y_hat : array, optional
        Estimated potential outcome of shape (n, p, a, 2). The default is None.
    **kwargs : dict
        Additional arguments to pass to the model.

    Returns
    -------    
    Y_hat : array
        Estimated potential outcome under control.
    pi_hat : array
        Estimated propensity score.
    '''
    func_ps, params_ps = _get_func_ps(ps_model, **kwargs)
    params_glm = _filter_params(fit_glm, kwargs)

    if verbose:
        pprint.pprint(params_ps)
        pprint.pprint(params_glm)
    
    if K>1:
        # Initialize KFold cross-validator
        kf = KFold(n_splits=K, random_state=0, shuffle=True)
        folds = kf.split(X)
    else:
        folds = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]

    # Initialize lists to store results
    fit_pi = True if pi_hat is None else False
    pi_hat = np.zeros_like(A, dtype=float) if fit_pi else pi_hat
    Y_hat = np.zeros((Y.shape[0],Y.shape[1],A.shape[1],2), dtype=float)

    # Perform cross-fitting
    for train_index, test_index in folds:
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        XA_train, XA_test = X_A[train_index], X_A[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if fit_pi:
            i_ctrl = (np.sum(A_train, axis=1) == 0.)

            pi = np.zeros_like(A_test, dtype=float)
            for j in range(A.shape[1]):
                i_case = (A_train[:,j] == 1.)
                i_cells = i_ctrl | i_case

                if ps_model=='logistic' and XA_train.shape[1]==1 and np.all(XA_train==1):
                    prob = np.sum(i_case)/np.sum(i_cells)
                    pi[A_train[:,j] == 1., j] = prob
                    pi[A_train[:,j] == 0., j] = 1 - prob
                else:
                    pi[:,j] = func_ps(XA_train[i_cells], A_train[i_cells][:,j], XA_test)

        # Fit GLM on training data and predict on test data
        res = fit_glm(Y_train, X_train, A_train, family=family, alpha=glm_alpha,
            impute=X_test, **params_glm)
        
        # Store results
        if fit_pi:
            pi_hat[test_index] = pi

        Y_hat[test_index,:,:,0] = res[1][0]
        Y_hat[test_index,:,:,1] = res[1][1]

    pi_hat = np.clip(pi_hat, 0.01, 0.99)
    Y_hat = np.clip(Y_hat, None, 1e5)
    return Y_hat, pi_hat





def AIPW_mean(Y, A, mu, pi, positive=False):
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
    positive : bool, optional
        Whether to restrict the pseudo-outcome to be positive.

    Returns
    -------
    tau : array
        A point estimate of the expected potential outcome of shape (p, a, 2).
    pseudo_y : array
        Pseudo-outcome of shape (n, p, a, 2).
    '''
    
    weight = A / pi
    weight = weight[:, None, ...]
    Y = Y[:, :, None, None]

    pseudo_y = weight * (Y - mu) + mu
    
    if positive:
        pseudo_y = np.clip(pseudo_y, 0, None)

    tau = np.mean(pseudo_y, axis=0)

    return tau, pseudo_y
    









from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn_ensemble_cv import reset_random_seeds, Ensemble, ECV
from sklearn.tree import DecisionTreeRegressor

def fit_rf(X, y, X_test=None, sample_weight=None, M=100, M_max=1000,
    # fixed parameters for bagging regressor
    kwargs_ensemble={'verbose':1},
    # fixed parameters for decision tree
    kwargs_regr={'min_samples_leaf': 3}, # 'min_samples_split': 10, 'max_features':'sqrt'
    # grid search parameters
    grid_regr = {'max_depth': [11]},
    grid_ensemble = {'random_state': 0}, #'max_samples':np.linspace(0.25, 1., 4)
    ):

    # Validate integer parameters
    M = int(M)
    M_max = int(M_max)
    # for kwargs in [kwargs_regr, kwargs_ensemble, grid_regr, grid_ensemble]:
    #     for param in kwargs:
    #         if param in ['max_depth', 'random_state', 'max_leaf_nodes'] and isinstance(kwargs[param], float):
    #             kwargs[param] = int(kwargs[param])

    # Make sure y is 2D
    y = y.reshape(-1, 1) if y.ndim == 1 else y

    # Run ECV
    res_ecv, info_ecv = ECV(
        X, y, DecisionTreeRegressor, grid_regr, grid_ensemble, 
        kwargs_regr, kwargs_ensemble, 
        M=M, M0=M, M_max=M_max, return_df=True
    )

    # Replace the in-sample best parameter for 'n_estimators' with extrapolated best parameter
    info_ecv['best_params_ensemble']['n_estimators'] = info_ecv['best_n_estimators_extrapolate']

    # Fit the ensemble with the best CV parameters
    regr = Ensemble(
        estimator=DecisionTreeRegressor(**info_ecv['best_params_regr']),
        **info_ecv['best_params_ensemble']).fit(X, y, sample_weight=sample_weight)
        
    # Predict
    if X_test is None:
        X_test = X
    return regr.predict(X_test).reshape(-1, y.shape[1])



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
        sample_weight = np.ones(y.shape[0])
        class_weight =  len(y) / (2 * np.bincount(y.astype(int)))  
        for a in range(2):
            sample_weight[y == a] = class_weight[a]     
        return fit_rf(X[i_cells], y[i_cells], sample_weight=sample_weight[i_cells], *args, **kwargs)

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