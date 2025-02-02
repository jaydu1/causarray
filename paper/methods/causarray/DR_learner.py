import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy as sp
from scipy.stats import norm
from causarray.DR_estimation import AIPW_mean, cross_fitting, est_var, fit_ps
from causarray.gcate_glm import loess_fit, ls_fit
from causarray.DR_inference import multiplier_bootstrap, step_down, augmentation
from causarray.utils import reset_random_seeds, pprint, tqdm, comp_size_factor, _filter_params
from statsmodels.stats.multitest import multipletests, fdrcorrection



def LFC(
    Y, W, A, W_A=None, family='nb', offset=False,
    B=1000, alpha=0.05, c=0.1, 
    Y_hat_0=None, Y_hat_1=None, pi=None, cross_est=False, 
    thres_min=1e-4, thres_diff=1e-6, fdx=False,
    verbose=False, **kwargs):
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
    verbose : bool
        Whether to print the model information.
    **kwargs : dict
        Additional arguments to pass to fit_glm.
    
    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(0)

    kwargs = {k:v for k,v in kwargs.items() if k not in 
        ['kwargs_ls_1', 'kwargs_ls_2', 'kwargs_es_1', 'kwargs_es_2', 'c1', 'num_d']
    }
    if len(A.shape)==1:
        A = A[:,None]

    if verbose:
        d_A = W.shape[1] if W_A is None else W_A.shape[1]
        pprint.pprint('Estimating LFC...')
        pprint.pprint({'estimands':'LFC','n':Y.shape[0],'p':Y.shape[1],'d':W.shape[1], 'd_A':d_A, 'a':A.shape[1]}, compact=True)
    
    n = W.shape[0]
    p = Y.shape[1]

    if offset is not None and offset is not False:
        if type(offset)==bool and offset is True:
            size_factors = comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs))
            offset = np.log(size_factors)
        else:
            size_factors = np.exp(offset)
    else:
        offset = None
        size_factors = np.ones(n)

    Y = Y.astype('float')
    if Y_hat_0 is None or Y_hat_1 is None:    
        pi, Y_hat_0, Y_hat_1 = cross_fitting(Y, W, A, W_A, family=family, offset=offset, verbose=verbose, **kwargs)
    elif pi is None:
        pi = fit_ps(W if W_A is None else W_A, A, verbose=verbose, **kwargs)
    pi = pi.reshape(*A.shape)

    # point estimation of the treatment effect
    _, eta_0s = AIPW_mean(Y, 1-A, np.clip(Y_hat_0, None, 1e5), 1-pi, pseudo_outcome=True, positive=True)
    _, eta_1s = AIPW_mean(Y, A, np.clip(Y_hat_1, None, 1e5), pi, pseudo_outcome=True, positive=True)
    eta_0s /= size_factors[:,None,None]
    eta_1s /= size_factors[:,None,None]

    i_ctrl = (np.sum(A, axis=1) == 0.)

    res = []
    iters =  range(A.shape[1]) if A.shape[1]==1 else tqdm(range(A.shape[1]))
    for j in iters:
        i_case = (A[:,j] == 1.)
        i_cells = i_ctrl | i_case
        n_cells = np.sum(i_cells)
        eta_0, eta_1 = eta_0s[i_cells,:,j], eta_1s[i_cells,:,j]
        tau_0, tau_1 = np.mean(eta_0, axis=0), np.mean(eta_1, axis=0)
        # eta_0, eta_1, tau_0, tau_1 = eta_0s[:,:,j], eta_1s[:,:,j], tau_0s[:,j], tau_1s[:,j]
        
        if cross_est:
            tau_estimate, eta, theta_var = est_var(eta_0, eta_1, thres_diff=thres_diff, log=True)
            theta_var = 2*theta_var
        else:
            tau_1 = np.clip(tau_1, thres_diff, None)
            tau_0 = np.clip(tau_0, thres_diff, None)

            tau_estimate = np.log(tau_1/tau_0)
            eta = eta_1 / tau_1[None,:] -  eta_0 / tau_0[None,:]
            theta_var = np.var(eta, axis=0, ddof=1)
        
        # filter out low-expressed genes
        idx = ~ ((np.maximum(tau_0,tau_1)<thres_min) & ((tau_1-tau_0)<thres_diff))
        tau_estimate[~idx] = 0.; eta[:,~idx] = 0.; theta_var[~idx] = np.inf

        sqrt_theta_var = np.sqrt((theta_var + 1e-3)/ n_cells)
        tvalues_init = tau_estimate / sqrt_theta_var

        # Multiple testing procedure
        if fdx:
            id_test = theta_var>=1e-4
            z_init = multiplier_bootstrap(eta, theta_var, B)
            z_init[:,~id_test] = 0.
            tvalues = tvalues_init.copy()
            tvalues[~id_test] = 0.
            V, tvalues, z = step_down(tvalues, z_init, alpha)

            V[(~id_test) & (np.abs(tau_estimate)>0.1)] = 1.
            V = augmentation(V, tvalues, c)
        else:
            V = np.zeros(p)

        # BH correction
        pvals = np.full(tvalues_init.shape, np.nan)
        qvals = np.full(tvalues_init.shape, np.nan)
        pvals[idx] = sp.stats.norm.sf(np.abs(tvalues_init[idx]))*2
        qvals[idx] = multipletests(pvals[idx], alpha=0.05, method='fdr_bh')[1]


        idx = ~np.isinf(theta_var)
        med = np.nanmedian(tvalues_init[idx])
        mad = sp.stats.median_abs_deviation(tvalues_init[idx], scale="normal", nan_policy='omit')
        tvalues_init_adj = (tvalues_init - med) / mad
        
        # BH correction
        pvals_adj = np.full(tvalues_init.shape, np.nan)
        qvals_adj = np.full(tvalues_init.shape, np.nan)
        pvals_adj[idx] = sp.stats.norm.sf(np.abs(tvalues_init_adj[idx]))*2
        qvals_adj[idx] = multipletests(pvals_adj[idx], alpha=0.05, method='fdr_bh')[1]
        
        df_res = pd.DataFrame({
            'tau': tau_estimate,
            'std': sqrt_theta_var,
            'stat': tvalues_init,
            'rej': V,
            'pvalue': pvals,
            'padj': qvals,
            'pvalue_emp_null_adj': pvals_adj,
            'padj_emp_null_adj': qvals_adj,            
            })
        if A.shape[1]>1:
            df_res['trt'] = j
        res.append(df_res)
    df_res = pd.concat(res, axis=0).reset_index(drop=True)
    estimation = {**{'pi':pi, 'Y_hat_0':Y_hat_0, 'Y_hat_1':Y_hat_1, 
        'W_A':W_A,
        'offset':offset, 'size_factors':size_factors}, **kwargs}
    return df_res, estimation





def VIM(eta, X, id_covs, **kwargs):
    '''
    Estimate the variable importance measure (VIM) using AIPW.

    Parameters
    ----------
    eta : array
        n x p matrix of influence function values.
    '''
    if len(X.shape)==1:
        X = X[:,None]

    n, p = eta.shape
    d = X.shape[1]
    if id_covs is None:
        id_covs = range(d)
    if np.isscalar(id_covs):
        id_covs = range(id_covs)

    n_covs = len(id_covs)

    emp_VTE = (eta - np.mean(eta, axis=0, keepdims=True))**2
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
        # regression eta on X to get predicted values
        if np.all(np.modf(X[:,i:i+1])[0] == 0):
            CATE[j], CATE_lower[j], CATE_upper[i] = ls_fit(eta, X[:,i], **kwargs)
        else:
            CATE[j], CATE_lower[j], CATE_upper[j] = loess_fit(eta, X[:,i], **kwargs)
        # compute the variance of treatment effect        
        _emp_CVTE = (eta - CATE[j])**2
        _CVTE = np.nanmean(_emp_CVTE, axis=0)
        emp_CVTE[j] = _emp_CVTE
        CVTE[j] = _CVTE

        VIM_mean[j] = _CVTE/VTE - 1
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