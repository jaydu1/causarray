import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
from scipy.stats import norm
from causarray.DR_estimation import AIPW_mean, cross_fitting, est_var, fit_ps
from causarray.DR_inference import multiplier_bootstrap, step_down, augmentation
from causarray.utils import reset_random_seeds
from statsmodels.stats.multitest import multipletests



def ATE(
    Y, W, A, W_A=None, 
    B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the average treatment effects (ATEs) using AIPW.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    A : array
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

    if len(A.shape)>1:
        A = A[:,0]

    n = W.shape[0]
    p = Y.shape[1]

    pi, Y_hat_0, Y_hat_1 = cross_fitting(Y, W, A, W_A, family=family, **kwargs)
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Y, 1-A, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Y, A, Y_hat_1, pi, pseudo_outcome=True)

    tau_estimate = tau_1 - tau_0
    eta = eta_1 - eta_0  - tau_estimate[None, :]

    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta, theta_var, B)
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
    Y, W, A, W_A=None, B=1000, alpha=0.05, c=0.01, family='poisson', **kwargs):
    '''
    Estimate the standardized average treatment effects (SATEs) using AIPW.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    A : array
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

    if len(A.shape)>1:
        A = A[:,0]

    n = W.shape[0]
    p = Y.shape[1]

    pi, Y_hat_0, Y_hat_1 = cross_fitting(Y, W, A, W_A, family=family, **kwargs)
    Y2_hat_0 = Y_hat_0**2
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Y, 1-A, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Y, A, Y_hat_1, pi, pseudo_outcome=True)
    _, eta_2 = AIPW_mean(Y**2, 1-A, Y2_hat_0, 1-pi, pseudo_outcome=True)

    idx = np.mean(eta_2, axis=0) - np.mean(eta_0, axis=0)**2 <= 0.
    print(np.sum(idx))
    sd = np.sqrt(np.maximum(
        np.mean(eta_2, axis=0) - np.mean(eta_0, axis=0)**2, 1e-6))[None,:]
    tau_estimate = np.mean((eta_1 - eta_0) / sd, axis=0)
    tau_estimate[idx] = 0.
    eta = (eta_1 - eta_0) / sd - tau_estimate[None,:] * (
        eta_2 + np.mean(eta_2, axis=0, keepdims=True) - 
        2 * np.mean(eta_0, axis=0, keepdims=True) * eta_0)/ (2 * sd**2)
    
    theta_var = np.var(eta, axis=0, ddof=1) 
    sqrt_theta_var = np.sqrt(theta_var)

    # standardized treatment effect
    tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var

    # Multiple testing procedure
    z_init = multiplier_bootstrap(eta, theta_var, B)
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



def FC(
    Y, W, A, W_A=None, B=1000, alpha=0.05, c=0.01, family='poisson', 
    Y_hat_0=None, Y_hat_1=None, cross_est=False, **kwargs):
    '''
    Estimate the fold changes of treatment effects (FCs) using AIPW.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    A : array
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

    if len(A.shape)>1:
        A = A[:,0]
        
    n = W.shape[0]
    p = Y.shape[1]

    if Y_hat_0 is None or Y_hat_1 is None:    
        pi, Y_hat_0, Y_hat_1 = cross_fitting(Y, W, A, W_A, family=family, **kwargs)
    else:
        pi = fit_ps(W, A, **kwargs)
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Y, 1-A, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Y, A, Y_hat_1, pi, pseudo_outcome=True)

    # idx = (tau_0 <= 0.)
    # print(np.sum(idx))
    if cross_est:
        tau_estimate, eta, theta_var = est_var(eta_0, eta_1)
        sqrt_theta_var = np.sqrt(theta_var)
        tvalues_init = np.sqrt(n/2) * (tau_estimate) / sqrt_theta_var
    else:
        tau_estimate = tau_1/tau_0 - 1
        # tau_estimate[idx] = 0.
        eta = (eta_1 - eta_0) / tau_0[None,:] - eta_0 * (tau_estimate / tau_0)[None,:]
        theta_var = np.var(eta, axis=0, ddof=1) #est_var(eta_0, eta_1)##    

        sqrt_theta_var = np.sqrt(theta_var)
        # standardized treatment effect
        tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var    

    # Multiple testing procedure
    id_test = theta_var>=1e-4
    z_init = multiplier_bootstrap(eta, theta_var, B)
    z_init[:,~id_test] = 0.
    tvalues = tvalues_init.copy()
    tvalues[~id_test] = 0.
    V, tvalues, z = step_down(tvalues, z_init, alpha)

    V[(~id_test) & (np.abs(tau_estimate)>0.1)] = 1.
    V = augmentation(V, tvalues, c)

    # z_init = multiplier_bootstrap(eta, theta_var, B)
    # V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    # V = augmentation(V, tvalues, c)

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



def LFC(
    Y, W, A, W_A=None, B=1000, alpha=0.05, c=0.01, family='poisson', 
    Y_hat_0=None, Y_hat_1=None, cross_est=False, **kwargs):
    '''
    Estimate the log-fold chanegs of treatment effects (LFCs) using AIPW.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    A : array
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

    if len(A.shape)>1:
        A = A[:,0]
        
    n = W.shape[0]
    p = Y.shape[1]

    if Y_hat_0 is None or Y_hat_1 is None:    
        pi, Y_hat_0, Y_hat_1 = cross_fitting(Y, W, A, W_A, family=family, **kwargs)
    else:
        pi = fit_ps(W, A, **kwargs)
    
    # point estimation of the treatment effect
    tau_0, eta_0 = AIPW_mean(Y, 1-A, Y_hat_0, 1-pi, pseudo_outcome=True)
    tau_1, eta_1 = AIPW_mean(Y, A, Y_hat_1, pi, pseudo_outcome=True)

    # idx = (tau_0 <= 0.)
    # print(np.sum(idx))
    if cross_est:
        tau_estimate, eta, theta_var = est_var(eta_0, eta_1)
        sqrt_theta_var = np.sqrt(theta_var)
        tvalues_init = np.sqrt(n/2) * (tau_estimate) / sqrt_theta_var
    else:
        tau_estimate = np.log(tau_1/(tau_0+1e-8)+1e-8)
        # tau_estimate[idx] = 0.
        eta = eta_1 / tau_1[None,:] -  eta_0 / tau_0[None,:]
        theta_var = np.var(eta, axis=0, ddof=1) #est_var(eta_0, eta_1)##    

        sqrt_theta_var = np.sqrt(theta_var)
        # standardized treatment effect
        tvalues_init = np.sqrt(n) * (tau_estimate) / sqrt_theta_var    

    # Multiple testing procedure
    id_test = theta_var>=1e-4
    z_init = multiplier_bootstrap(eta, theta_var, B)
    z_init[:,~id_test] = 0.
    tvalues = tvalues_init.copy()
    tvalues[~id_test] = 0.
    V, tvalues, z = step_down(tvalues, z_init, alpha)

    # V[(~id_test) & (np.abs(tau_estimate)>0.1)] = 1.
    V = augmentation(V, tvalues, c)

    # z_init = multiplier_bootstrap(eta, theta_var, B)
    # V, tvalues, z = step_down(tvalues_init, z_init, alpha)
    # V = augmentation(V, tvalues, c)

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