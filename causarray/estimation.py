import numpy as np
from scipy.optimize import root_scalar
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import QuantileRegressor, LogisticRegression
from quantile_forest import RandomForestQuantileRegressor
from joblib import Parallel, delayed
import scipy as sp
from causarray.glm_test import fit_glm

n_jobs = 8


def density_estimate(y, x, **kwargs):
    from scipy.stats import nbinom, poisson, geom

    densities = {}
    likelihoods = {}

    mean = np.mean(y)
    var = np.var(y)
    y = np.round(y)
    likelihoods['poisson'] = poisson.logpmf(y, mean).sum()
    densities['poisson'] = sp.stats.poisson.pmf(x, mean)

    p = mean / var
    r = p * mean / (1-p)
    likelihoods['nbinom'] = nbinom.logpmf(y, r, p).sum()
    densities['nbinom'] = sp.stats.nbinom.pmf(x, r, p)

    p = 1 / mean
    likelihoods['geometric'] = geom.logpmf(y, p).sum()
    densities['geometric'] = sp.stats.geom.pmf(x, p)

    best_fit = max(likelihoods, key=lambda x: likelihoods[x])

    return densities[best_fit]



def AIPW_mean(y, A, mu, pi, pseudo_outcome=False):
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
    tau = np.mean(pseudo_y, axis=0)

    if pseudo_outcome:
        return tau, pseudo_y
    else:
        return tau
    



def AIPW_quantile(y, A, tau_init, density, nu, pi, q, pseudo_outcome=False):
    '''
    Augmented inverse probability weighted estimator (AIPW)

    Parameters
    ----------
    y : array
        Outcomes.
    A : array
        Binary treatment indicator.
    tau_init : array
        The initial estimate of the quantile of the potential outcome.
    density : array
        The estimated density of the initial estimate of the quantile of the potential outcome.
    nu : array
        The estimated conditional cumulative distribution function P(Y(a) <= tau_init | W, A).
    pi : array
        The estimated propensity score.
    q : float
        Quantile to be computed (e.g., q = 0.5 for the median.)
    pseudo_outcome : bool, optional
        Whether to return the pseudo-outcome. The default is False.
    
    Returns
    -------
    tau : array
        A point estimate of the quantile of the potential outcome.
    pseudo_y : array
        Pseudo-outcome if `pseudo_outcome = True`.
    '''
    weight = A / pi
    
    pseudo_y = tau_init + (weight * (nu - (y <= tau_init)) - (nu - q)) / density
    tau = np.mean(pseudo_y)

    if pseudo_outcome:
        return tau, pseudo_y
    else:
        return tau








def IPW_quantile(y, A, pi, q, pseudo_outcome=False, **kwargs):
    """
    Inverse probability weighted estimator (IPW)

    Parameters
    ----------
    y : array
        Outcomes.
    A : array
        Binary treatment indicator.
    Q : array
        Conditional outcome distribution estimate. It is an n x p matrix, where
        p columns represent p conditional quantiles evenly spaced from 0 to 1.
    pi : array
        Propensity score.
    q : float
        Quantile to be computed (e.g., q = 0.5 for the median.)
    pseudo_outcome : bool, optional
        Whether to return the pseudo-outcome. The default is False.
    **kwargs :
        Keyword arguments for kernel density estimation.

    Returns
    -------
    tau : array
        A point estimate of the quantile of the potential outcome.
    pseudo_y : array
        Pseudo-outcome if `pseudo_outcome = True`.        
    """
    weight = A / pi

    def D(chiq):
        return np.mean(weight * ((y <= chiq) - q)) 

    tau = root_scalar(D, bracket=[y.min()-1000, y.max()+1000]).root

    if pseudo_outcome:
        pseudo_y = -1. / density_estimate(y, np.round(np.array([tau])))
        return tau, pseudo_y
    else:
        return tau


def gaussian_kernel(x, y):
    h = 0.9 * np.minimum(np.std(y), sp.stats.iqr(y)/1.35) * len(y) ** (-1 / 5)

    return np.exp(-((x - y) / h) ** 2 / 2) / h / np.sqrt(2 * np.pi)


def AIPW_quantile(y, X, A, mu, pi, q, pseudo_outcome=False, **kwargs):
    """
    Augmented inverse probability weighted estimator (AIPW)

    Parameters
    ----------
    y : array
        Outcomes.
    A : array
        Binary treatment indicator.
    nu : array
        Conditional outcome distribution estimate. It is an n x p matrix, where
        p columns represent p conditional quantiles evenly spaced from 0 to 1.
    pi : array
        Propensity score.
    q : float
        Quantile to be computed (e.g., q = 0.5 for the median.)
    pseudo_outcome : bool, optional
        Whether to return the pseudo-outcome. The default is False.
    **kwargs :
        Keyword arguments for kernel density estimation.

    Returns
    -------
    tau : array
        A point estimate of the quantile of the potential outcome.
    pseudo_y : array
        Pseudo-outcome if `pseudo_outcome = True`.        
    """

    tau_init = IPW_quantile(y, A, pi, q)
    weight = A / pi
    # density = density_estimate(y, np.round(np.array([tau_init])))
    # density = density_estimate(mu, np.round(np.array([tau_init])))
    density_y = gaussian_kernel(tau_init, y)
    density_mu = gaussian_kernel(tau_init, mu)
    density = np.mean(weight * (density_y - density_mu) + density_mu)

    if 0<np.mean(y<=tau_init)<1:
        clf = LogisticRegression(random_state=0, fit_intercept=False).fit(np.c_[X, A[:,None]], (y<=tau_init))
        nu = clf.predict_proba(np.c_[X, np.ones_like(A)[:,None]])[:,1] - q
    else:
        nu = np.mean(y<=tau_init) - q
    # nu = sp.stats.poisson.cdf(tau_init, mu=mu) - q
    pseudo_y = tau_init + (weight * (nu - ((y <= tau_init) - q) ) - nu) / density
    tau = np.mean(pseudo_y)

    if pseudo_outcome:
        return tau, pseudo_y
    else:
        return tau



def fit_qr(Y, X, A, pi, lower=0.25, upper=0.75, family='poisson', **kwargs):
    '''
    Fit quantile regression to each column of Y, with covariate X and treatment A.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes
    X : array
        n x d matrix of covariates
    A : array
        n x 1 vector of treatments
    pi : array
        n x 1 vector of propensity scores
    **kwargs : dict
        additional arguments to pass to QuantileRegressor
    
    Returns
    -------
    B : array
        p x d matrix of coefficients
    Yhat : array or tuple
        n x p matrix of predicted values or tuple of predicted potential outcomes
    '''
    d = X[:,:].shape[1]

    def fit_qr_j(j):
        rho_0, eta_0 = AIPW_quantile(Y[:,j], X, 1-A, Y_hat_0[:,j], 1-pi, 0.5, pseudo_outcome=True)
        rho_1, eta_1 = AIPW_quantile(Y[:,j], X, A, Y_hat_1[:,j], pi, 0.5, pseudo_outcome=True)
        if lower is None:
            rho_0_lower, eta_0_lower = np.nan, np.nan
        else:
            rho_0_lower, eta_0_lower = AIPW_quantile(Y[:,j], X, 1-A, Y_hat_0[:,j], 1-pi, lower, pseudo_outcome=True)
        if upper is None:
            rho_0_upper, eta_0_upper = np.nan, np.nan
        else:
            rho_0_upper, eta_0_upper = AIPW_quantile(Y[:,j], X, 1-A, Y_hat_0[:,j], 1-pi, upper, pseudo_outcome=True)

        return rho_0, rho_1, rho_0_upper-rho_0_lower, eta_0, eta_1, eta_0_upper-eta_0_lower

    # generalized linear regression
    # quantiles_pred = [lower, 0.5, upper]
    Y_hat_0, Y_hat_1 = fit_glm(Y, X, A, family=family, impute=True)[1]
    # Q_hat_0 = np.r_[[sp.stats.poisson.ppf(i, mu=Y_hat_0) for i in quantiles_pred]]
    # Q_hat_1 = np.r_[[sp.stats.poisson.ppf(i, mu=Y_hat_1) for i in quantiles_pred]]

    with Parallel(n_jobs=n_jobs, verbose=0, timeout=99999) as parallel:
        res = parallel(delayed(fit_qr_j)(j) for j in range(Y.shape[1]))

    rho_0, rho_1, iqr, eta_0, eta_1, eta_iqr = list(zip(*res))
    rho_0 = np.array(rho_0)
    rho_1 = np.array(rho_1)
    iqr = np.array(iqr)
    
    eta_0 = np.array(eta_0).T
    eta_1  = np.array(eta_1).T
    eta_iqr = np.array(eta_iqr).T
    
    return rho_0, rho_1, iqr, eta_0, eta_1, eta_iqr


# def AIPW_quantile(y, A, Q, pi, q, pseudo_outcome=False, **kwargs):
#     """
#     Augmented inverse probability weighted estimator (AIPW)

#     Parameters
#     ----------
#     y : array
#         Outcomes.
#     A : array
#         Binary treatment indicator.
#     Q : array
#         Conditional outcome distribution estimate. It is an n x p matrix, where
#         p columns represent p conditional quantiles evenly spaced from 0 to 1.
#     pi : array
#         Propensity score.
#     q : float
#         Quantile to be computed (e.g., q = 0.5 for the median.)
#     pseudo_outcome : bool, optional
#         Whether to return the pseudo-outcome. The default is False.
#     **kwargs :
#         Keyword arguments for kernel density estimation.

#     Returns
#     -------
#     tau : array
#         A point estimate of the quantile of the potential outcome.
#     pseudo_y : array
#         Pseudo-outcome if `pseudo_outcome = True`.        
#     """
#     weight = A / pi

#     def D(chiq):
#         nu = np.mean((Q <= chiq), axis=1)
#         return np.mean(weight * ((y <= chiq) - nu)) + np.mean(nu) - q

#     tau = root_scalar(D, bracket=[y.min()-1000, y.max()+1000]).root
#     if pseudo_outcome:        
#         nu = np.mean((Q <= tau), axis=1)

#         density = density_estimate(y, np.round(np.array([tau])))
#         # kde = KernelDensity(**kwargs).fit(y[:,None])
#         # density = np.exp(kde.score_samples(np.array([[tau]])))
#         pseudo_y = -1. / density * (weight * ((y <= tau) - nu) + nu - q)
#         return tau, pseudo_y
#     else:
#         return tau


# def fit_qr(Y, X, A, pi, lower=0.25, upper=0.75, family='poisson', **kwargs):
#     '''
#     Fit quantile regression to each column of Y, with covariate X and treatment A.

#     Parameters
#     ----------
#     Y : array
#         n x p matrix of outcomes
#     X : array
#         n x d matrix of covariates
#     A : array
#         n x 1 vector of treatments
#     pi : array
#         n x 1 vector of propensity scores
#     **kwargs : dict
#         additional arguments to pass to QuantileRegressor
    
#     Returns
#     -------
#     B : array
#         p x d matrix of coefficients
#     Yhat : array or tuple
#         n x p matrix of predicted values or tuple of predicted potential outcomes
#     '''
#     d = X[:,:].shape[1]

#     def fit_qr_j(j):
#         quantiles_pred = np.linspace(0.01, 0.99, 99).tolist()

#         # random forest quantile regression
#         qrf = RandomForestQuantileRegressor().fit(np.c_[X,A], Y[:,j])
#         Q_hat_0 = qrf.predict(np.c_[X,np.zeros_like(A)], quantiles=quantiles_pred)
#         Q_hat_1 = qrf.predict(np.c_[X,np.ones_like(A)], quantiles=quantiles_pred)

#         # quantile regression
#         # Q_hat_0 = np.zeros((X.shape[0], len(quantiles_pred)))
#         # Q_hat_1 = np.zeros((X.shape[0], len(quantiles_pred)))
#         # for quantile in quantiles_pred:
#         #     qr = QuantileRegressor(
#         #         quantile=quantile, fit_intercept=False, solver='highs', alpha=0., **kwargs
#         #         ).fit(np.c_[X,A], Y[:,j])
#         #     Q_hat_0[:,0] = qr.predict(np.c_[X,np.zeros_like(A)])
#         #     Q_hat_1[:,0] = qr.predict(np.c_[X,np.ones_like(A)])

        
#         rho_0, eta_0 = AIPW_quantile(Y[:,j], 1-A, Q_hat_0, 1-pi, 0.5, pseudo_outcome=True)
#         rho_1, eta_1 = AIPW_quantile(Y[:,j], A, Q_hat_1, pi, 0.5, pseudo_outcome=True)
#         rho_0_lower, eta_0_lower = AIPW_quantile(Y[:,j], 1-A, Q_hat_0, 1-pi, lower, pseudo_outcome=True)
#         rho_0_upper, eta_0_upper = AIPW_quantile(Y[:,j], 1-A, Q_hat_0, 1-pi, upper, pseudo_outcome=True)

#         # rho_0, eta_0 = AIPW_quantile(Y[:,j], 1-A, Q_hat_0[:,:,j].T, 1-pi, 0.5, pseudo_outcome=True)
#         # rho_1, eta_1 = AIPW_quantile(Y[:,j], A, Q_hat_1[:,:,j].T, pi, 0.5, pseudo_outcome=True)
#         # rho_0_lower, eta_0_lower = AIPW_quantile(Y[:,j], 1-A, Q_hat_0[:,:,j].T, 1-pi, lower, pseudo_outcome=True)
#         # rho_0_upper, eta_0_upper = AIPW_quantile(Y[:,j], 1-A, Q_hat_0[:,:,j].T, 1-pi, upper, pseudo_outcome=True)

#         return rho_0, rho_1, rho_0_upper-rho_0_lower, eta_0, eta_1, eta_0_upper-eta_0_lower

#     # generalized linear regression
#     # quantiles_pred = np.linspace(0.01, 0.99, 99).tolist()
#     # Y_hat_0, Y_hat_1 = fit_glm(Y, X, A, family=family, impute=True)[1]
#     # Q_hat_0 = np.r_[[sp.stats.poisson.ppf(i, mu=Y_hat_0) for i in quantiles_pred]]
#     # Q_hat_1 = np.r_[[sp.stats.poisson.ppf(i, mu=Y_hat_1) for i in quantiles_pred]]

#     with Parallel(n_jobs=n_jobs, verbose=0, timeout=99999) as parallel:
#         res = parallel(delayed(fit_qr_j)(j) for j in range(Y.shape[1]))

#     rho_0, rho_1, iqr, eta_0, eta_1, eta_iqr = list(zip(*res))
#     rho_0 = np.array(rho_0)
#     rho_1 = np.array(rho_1)
#     iqr = np.array(iqr)
    
#     eta_0 = np.array(eta_0).T
#     eta_1  = np.array(eta_1).T
#     eta_iqr = np.array(eta_iqr).T
    
#     return rho_0, rho_1, iqr, eta_0, eta_1, eta_iqr        



from sklearn.model_selection import KFold

def cross_fitting(Yg, W, D, family, K=3, **kwargs):
    # Get the list of valid parameters for LogisticRegression
    valid_params = LogisticRegression().get_params().keys()
    # Remove keys in kwargs that are not valid parameters for LogisticRegression
    valid_params = {k: v for k, v in kwargs.items() if k in valid_params}

    # Initialize KFold cross-validator
    kf = KFold(n_splits=K, random_state=0, shuffle=True)

    # Initialize lists to store results
    pi_list = np.zeros(W.shape[0])
    Y_hat_0_list = np.zeros_like(Yg)
    Y_hat_1_list = np.zeros_like(Yg)

    # Perform cross-fitting
    for train_index, test_index in kf.split(W):
        # Split data
        W_train, W_test = W[train_index], W[test_index]
        D_train, D_test = D[train_index], D[test_index]
        Yg_train, Yg_test = Yg[train_index], Yg[test_index]

        # Fit model on training data
        clf = LogisticRegression(random_state=0, fit_intercept=False, **valid_params).fit(W_train, D_train)

        # Predict on test data
        pi = clf.predict_proba(W_test)[:,1]

        # Fit GLM on training data and predict on test data
        res = fit_glm(Yg_train, W_train, D_train, family=family, impute=W_test)
        
        # Store results
        pi_list[test_index] = pi
        Y_hat_0_list[test_index] = res[1][0]
        Y_hat_1_list[test_index] = res[1][1]

    return pi_list, Y_hat_0_list, Y_hat_1_list