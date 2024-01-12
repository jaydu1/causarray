import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from causarray.estimation import fit_qr, AIPW_mean
from causarray.glm_test import fit_glm
from causarray.inference import multiplier_bootstrap, step_down, augmentation
from causarray.utils import reset_random_seeds
from statsmodels.stats.multitest import multipletests



def ATE(
    Yg, W, D, B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the average treatment effects (ATEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    **kwargs : dict
        Additional arguments to pass to fit_glm.
    
    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)

    n = W.shape[0]
    p = Yg.shape[1]

    Y_estimate = Yg

    # logistic regression for propensity score
    clf = LogisticRegression(random_state=0, fit_intercept=False, **kwargs).fit(W, D)
    pi = clf.predict_proba(W)[:,1]

    res = fit_glm(Yg, W, D, family=family, impute=True)

        # B_glm, _, tvals, pvals, _ = fit_glm(Yg**2, W, D, family='poisson', impute=False, offset=True)
        # mu_glm = np.mean(np.exp(np.c_[W,D] @ B_glm.T + np.log(np.sum(Yg**2, axis=1, keepdims=True))), axis=0)
        # disp_glm = (np.mean((Yg**2 - mu_glm[None,:])**2, axis=0) - mu_glm) / mu_glm**2
        # disp_glm = np.clip(disp_glm, 0.01, 100.)
        # res2 = fit_glm(Yg**2, W, D, 'nb', disp_glm, impute=True, offset=True)

    B_hat = res[0]
    Y_hat_0, Y_hat_1 = res[1]
    Y2_hat_0 = Y_hat_0**2
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Yg, 1-D, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Yg, D, Y_hat_1, pi, pseudo_outcome=True)

    eta = eta_1 - eta_0
    tau_estimate = tau_1 - tau_0
    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta - tau_estimate[None, :], theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)

    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df_res = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,
        'tvalues': tvalues,
        'rej': V,
        'pvals': pvals, 
        'qvals': qvals
        })

    return df_res



def SATE(
    Yg, W, D, B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the standardized average treatment effects (SATEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    **kwargs : dict
        Additional arguments to pass to fit_glm.
    
    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)

    n = W.shape[0]
    p = Yg.shape[1]

    Y_estimate = Yg

    # logistic regression for propensity score
    clf = LogisticRegression(random_state=0, fit_intercept=False, **kwargs).fit(W, D)
    pi = clf.predict_proba(W)[:,1]
    res = fit_glm(Yg, W, D, family=family, impute=True)

    B_hat = res[0]
    Y_hat_0, Y_hat_1 = res[1]
    Y2_hat_0 = Y_hat_0**2

    # res = fit_glm(Yg**2, W, D, family=family, impute=True)
    # Y2_hat_0, _ = res[1]
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Yg, 1-D, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Yg, D, Y_hat_1, pi, pseudo_outcome=True)
    _, eta_2 = AIPW_mean(Yg**2, 1-D, Y2_hat_0, 1-pi, pseudo_outcome=True)

    idx = np.mean(eta_2, axis=0) - np.mean(eta_0, axis=0)**2 <= 0.
    print(np.sum(idx))
    sd = np.sqrt(np.maximum(
        np.mean(eta_2, axis=0) - np.mean(eta_0, axis=0)**2, 1e-6))[None,:]
    eta = (eta_1 - eta_0) / sd
    tau_estimate = np.mean(eta, axis=0)
    tau_estimate[idx] = 0.
    theta_var = np.var(
        eta - tau_estimate[None,:] * (eta_2 + np.mean(eta_2, axis=0, keepdims=True) - 
        2 * np.mean(eta_0, axis=0, keepdims=True) * eta_0)/ (2 * sd**2)
        , axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta - tau_estimate[None, :], theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)

    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df_res = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,
        'tvalues': tvalues,
        'rej': V,
        'pvals': pvals, 
        'qvals': qvals
        })

    return df_res




def QTE(
    Yg, W, D, B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the quantile treatment effects (QTEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    **kwargs : dict
        Additional arguments to pass to fit_glm.

    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)
    
    n = W.shape[0]
    p = Yg.shape[1]

    # logistic regression for propensity score
    # Get the list of valid parameters for LogisticRegression
    valid_params = LogisticRegression().get_params().keys()
    # Remove keys in kwargs that are not valid parameters for LogisticRegression
    valid_params = {k: v for k, v in kwargs.items() if k in valid_params}
    clf = LogisticRegression(random_state=0, fit_intercept=False, **valid_params).fit(W, D)
    pi = clf.predict_proba(W)[:,1]

    # point estimation of the treatment effect
    rho_0, rho_1, iqr_0, eta_0, eta_1, eta_iqr = fit_qr(Yg, W, D, pi, **kwargs)
    
    tau_estimate = (rho_1 - rho_0)
    idx = (np.abs(tau_estimate) > 1e6)
    print(np.sum(idx))
    tau_estimate[idx] = 0.
    eta = (eta_1 - eta_0)
    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta - tau_estimate[None, :], theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)
    
    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df_res = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,

        'tvalues': tvalues,
        'rej': V,

        'pvals': pvals, 
        'qvals': qvals,
        })

    return df_res


def SQTE(
    Yg, W, D, B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the standardized quantile treatment effects (SQTEs) using AIPW.

    Parameters
    ----------
    Yg : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    D : array
        n x 1 vector of treatments.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        The augmentation parameter.
    **kwargs : dict
        Additional arguments to pass to fit_glm.

    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)
    
    n = W.shape[0]
    p = Yg.shape[1]

    # logistic regression for propensity score
    # Get the list of valid parameters for LogisticRegression
    valid_params = LogisticRegression().get_params().keys()
    # Remove keys in kwargs that are not valid parameters for LogisticRegression
    valid_params = {k: v for k, v in kwargs.items() if k in valid_params}
    clf = LogisticRegression(random_state=0, fit_intercept=False, **valid_params).fit(W, D)
    pi = clf.predict_proba(W)[:,1]

    # point estimation of the treatment effect
    rho_0, rho_1, iqr_0, eta_0, eta_1, eta_iqr = fit_qr(Yg, W, D, pi, **kwargs)
    
    
    
    tau_estimate = (rho_1 - rho_0)/iqr_0
    
    idx = np.isnan(iqr_0) | (np.abs(tau_estimate) > 1e6)
    print(np.sum(idx))

    tau_estimate[idx] = 0.
    eta = ((eta_1 - eta_0) - tau_estimate[None,:] * eta_iqr) / iqr_0
    eta[:,idx] = 0.
    # eta = (eta_1 - eta_0) / iqr_0
    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta - tau_estimate[None,:], theta_var, B)
    V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    V = augmentation(V, tvalues, c)

    # BH correction
    pvals = sp.stats.norm.sf(np.abs(tvalues_init))*2
    qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    df_res = pd.DataFrame({
        'tau_estimate': tau_estimate,
        'sqrt_theta_var': sqrt_theta_var,
        'tvalues_init': tvalues_init,
        'tvalues': tvalues,
        'rej': V,
        
        'pvals': pvals, 
        'qvals': qvals
        })

    return df_res

