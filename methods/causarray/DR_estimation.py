import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from causarray.gcate_glm import fit_glm
n_jobs = 8


def AIPW_mean(y, A, mu, pi, pseudo_outcome=False, positive=False):
    '''
    Augmented inverse probability weighted estimator (AIPW)

    Parameters
    ----------
    y : array
        Outcomes.
    A : array
        Binary treatment indicator.
    mu : array
        Conditional outcome distribution estimate.
    pi : array
        Propensity score.
    pseudo_outcome : bool, optional
        Whether to return the pseudo-outcome. The default is False.
    positive : bool, optional
        Whether to restrict the pseudo-outcome to be positive.

    Returns
    -------
    tau : array
        A point estimate of the expected potential outcome.
    pseudo_y : array
        Pseudo-outcome if `pseudo_outcome = True`.
    '''
    weight = A / pi
    if len(mu.shape)>1:
        weight = weight[:,None]
    pseudo_y = weight * (y - mu) + mu
    
    if positive:
        pseudo_y = np.clip(pseudo_y, 0, None)

    tau = np.mean(pseudo_y, axis=0)

    if pseudo_outcome:
        return tau, pseudo_y
    else:
        return tau
    

from sklearn.model_selection import KFold

def cross_fitting(
    Y, X, A, X_A=None, family='poisson', K=1, 
    ps_model='logistic', verbose=False, **kwargs):
    '''
    Cross-fitting for causal estimands.

    Parameters
    ----------
    Y : array
        Outcomes.
    X : array
        Covariates.
    A : array
        Binary treatment indicator.
    X_A : array, optional
        Covariates for the propensity score model. The default is None for using X.
    family : str, optional
        The family of the generalized linear model. The default is 'poisson'.
    K : int, optional
        The number of folds for cross-validation. The default is 1.
    **kwargs : dict
        Additional arguments to pass to the model.

    Returns
    -------
    pi_arr : array
        Propensity score.
    Y_hat_0_arr : array
        Estimated potential outcome under control.
    Y_hat_1_arr : array
        Estimated potential outcome under treatment.    
    '''
    if X_A is None:
        X_A = X

    if ps_model=='logistic':
        clf_ps = LogisticRegression
        kwargs['fit_intercept'] = False
    elif ps_model=='random_forest':
        clf_ps = RandomForestClassifier
        if 'min_samples_split' not in kwargs:
            kwargs['min_samples_split'] = 5
        
    # Get the list of valid parameters for LogisticRegression
    valid_params = clf_ps().get_params().keys()
    # Remove keys in kwargs that are not valid parameters for LogisticRegression
    valid_params = {k: v for k, v in kwargs.items() if k in valid_params}
    if verbose:
        pprint.pprint(valid_params)

    if 'disp_glm' in kwargs:
        disp_glm = kwargs['disp_glm']
    else:
        disp_glm = None
    
    if K>1:
        # Initialize KFold cross-validator
        kf = KFold(n_splits=K, random_state=0, shuffle=True)
        folds = kf.split(X)
    else:
        folds = [(np.arange(X.shape[0]), np.arange(X.shape[0]))]

    # Initialize lists to store results
    pi_arr = np.zeros(X.shape[0], dtype=float)
    Y_hat_0_arr = np.zeros_like(Y, dtype=float)
    Y_hat_1_arr = np.zeros_like(Y, dtype=float)

    # Perform cross-fitting
    for train_index, test_index in folds:
        # Split data
        X_train, X_test = X[train_index], X[test_index]
        XA_train, XA_test = X_A[train_index], X_A[test_index]
        A_train, A_test = A[train_index], A[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Estimate the proposensity score function on training data
        
        clf = clf_ps(random_state=0, **valid_params).fit(XA_train, A_train)

        # Predict on test data        
        pi = clf.predict_proba(XA_test)[:,1]

        # Fit GLM on training data and predict on test data
        res = fit_glm(Y_train, X_train, A_train, family=family, alpha=1e-4,
            disp_glm=disp_glm, impute=X_test)
        
        # Store results
        pi_arr[test_index] = pi
        
        Y_hat_0_arr[test_index] = res[1][0]
        Y_hat_1_arr[test_index] = res[1][1]

    pi_arr = np.clip(pi_arr, 0.01, 0.99)
    return pi_arr, Y_hat_0_arr, Y_hat_1_arr


def fit_ps(X, A,
    ps_model='random_forest', **kwargs):
    if ps_model=='logistic':
        clf_ps = LogisticRegression
        kwargs['fit_intercept'] = False
    elif ps_model=='random_forest':
        clf_ps = RandomForestClassifier
        if 'min_samples_split' not in kwargs:
            kwargs['min_samples_split'] = 5
        
    # Get the list of valid parameters for LogisticRegression
    valid_params = clf_ps().get_params().keys()
    # Remove keys in kwargs that are not valid parameters for LogisticRegression
    valid_params = {k: v for k, v in kwargs.items() if k in valid_params}

    clf = clf_ps(random_state=0, **valid_params).fit(X, A)

    # Predict on test data        
    pi = clf.predict_proba(X)[:,1]

    pi = np.clip(pi, 0.01, 0.99)
    return pi


from sklearn.model_selection import ShuffleSplit

def est_var(eta_0, eta_1, n_splits=10, log=False):
    eta = np.zeros_like(eta_0)
    tau = np.zeros(eta_0.shape[1])
    var = np.zeros(eta_0.shape[1])
    rs = ShuffleSplit(n_splits=n_splits, train_size=0.5, test_size=.5, random_state=0)
    for (train_i, test_i) in rs.split(eta_0):
        for (train_index, test_index) in [(train_i, test_i), (test_i, train_i)]:
            tau_1 = np.mean(eta_1[test_index], axis=0)
            tau_0 = np.mean(eta_0[test_index], axis=0)
            if log:
                tau_1 = np.clip(tau_1, 1e-8, None)
                tau_0 = np.clip(tau_0, 1e-8, None)
                tau_estimate = np.log(tau_1/tau_0)
            else:
                tau_estimate = tau_1/tau_0 - 1
            _eta = (eta_1 - eta_0)[train_index] / tau_0[None,:] - eta_0[train_index] * (tau_estimate / tau_0)[None,:]
            _var = np.var(_eta, axis=0, ddof=1)
            idx = np.isnan(tau_estimate)
            tau_estimate[idx] = 0.; _eta[:,idx] = 0.; _var = np.inf

            tau += tau_estimate
            eta[train_index] += _eta
            var += _var
    tau /= 2*n_splits
    eta /= n_splits
    var /= 2*n_splits
    return tau, eta, var

