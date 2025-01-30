import numpy as np
import pandas as pd
from causarray.DR_estimation import AIPW_mean, cross_fitting
from causarray.gcate_glm import loess_fit, ls_fit
from causarray.DR_inference import fdx_control, bh_correction
from causarray.utils import reset_random_seeds, pprint, tqdm, comp_size_factor, _filter_params




def compute_causal_estimand(
    estimand,
    Y, W, A, W_A=None, family='nb', offset=False,    
    Y_hat=None, pi_hat=None, eps_var=1e-3,
    fdx=False, fdx_B=1000, fdx_alpha=0.05, fdx_c=0.1,     
    verbose=False, random_state=0, **kwargs):
    '''
    Estimate the log-fold chanegs of treatment effects (LFCs) using AIPW.

    Parameters
    ----------
    estimand : function
        The causal estimand to estimate, it takes the estimated influence function values (eta_0, eta_1) 
        of ATE as input and returns the estimated treatment effect and the estimated influence function (tau, eta).
    Y : array
        n x p matrix of outcomes.
    W : array
        n x d matrix of covariates.
    A : array
        n x 1 vector of treatments.
    W_A : array, optional
        n x d_A matrix of covariates for treatment. If None, W is used.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    offset : array-like, optional
        Offset for the model.

    Y_hat : array, optional
        Predicted outcomes under treatment of shape (n, p, a, 2).
    pi_hat : array, optional
        Predicted propensity scores of shape (n, a).

    fdx : bool
        Whether to use FDX control, P(FDP > c) < alpha.
    fdx_B : int
        Number of bootstrap samples for FDX control.
    fdx_alpha : float
        The significance level for FDX control.
    fdx_c : float
        The augmentation parameter for FDX control.
    
    verbose : bool
        Whether to print the model information.
    **kwargs : dict
        Additional arguments to pass to fit_glm.
    
    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''
    reset_random_seeds(random_state)

    kwargs = {k:v for k,v in kwargs.items() if k not in 
        ['kwargs_ls_1', 'kwargs_ls_2', 'kwargs_es_1', 'kwargs_es_2', 'c1', 'num_d']
    }

    if isinstance(Y, pd.DataFrame):
        gene_names = Y.columns
        Y = Y.values
    else:
        gene_names = range(Y.shape[1])
    n, p = Y.shape

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

    Y = Y.astype('float')
    Y_hat, pi_hat = cross_fitting(Y, A, W, W_A, family=family, offset=offset, 
        Y_hat=Y_hat, pi_hat=pi_hat, random_state=random_state, verbose=verbose, **kwargs)
    pi_hat = pi_hat.reshape(*A.shape)

    # point estimation of the treatment effect
    _, etas = AIPW_mean(Y, np.stack([1-A, A], axis=-1), 
        Y_hat, np.stack([1-pi_hat, pi_hat], axis=-1), positive=True)

    # normalize the influence function values
    etas /= size_factors[:,None,None,None]


    i_ctrl = (np.sum(A, axis=1) == 0.)

    res = []
    iters = range(A.shape[1]) if A.shape[1]==1 else tqdm(range(A.shape[1]))
    for j in iters:
        i_case = (A[:,j] == 1.)
        i_cells = i_ctrl | i_case
        n_cells = np.sum(i_cells)        
        eta_est, tau_est, var_est = estimand(etas[i_cells,:,j], **kwargs)

        std_est = np.sqrt((var_est + eps_var)/ n_cells)
        tvalues_init = tau_est / std_est

        # Multiple testing procedure
        V = fdx_control(tau_est, var_est, tvalues_init, eta_est, fdx, fdx_B, fdx_alpha, fdx_c)

        # BH correction
        tvalues_init[np.isinf(var_est)] = np.nan
        pvals, qvals, pvals_adj, qvals_adj = bh_correction(tvalues_init)
        
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
        if A.shape[1]>1:
            df_res['trt'] = trt_names[j]
        res.append(df_res)
    df_res = pd.concat(res, axis=0).reset_index(drop=True)
    estimation = {**{'pi_hat':pi_hat, 'Y_hat':Y_hat, 'offset':offset, 'size_factors':size_factors}, **kwargs}
    return df_res, estimation


def LFC(
    Y, W, A, W_A=None, family='nb', offset=False,    
    Y_hat=None, pi_hat=None, cross_est=False, 
    thres_min=1e-4, thres_diff=1e-6, eps_var=1e-3,
    fdx=False, fdx_alpha=0.05, fdx_c=0.1,     
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
    W_A : array, optional
        n x d_A matrix of covariates for treatment. If None, W is used.
    family : str
        The distribution of the outcome. The default is 'poisson'.
    offset : array-like, optional
        Offset for the model.

    Y_hat : array, optional
        Predicted outcomes under treatment of shape (n, p, a, 2).
    pi_hat : array, optional
        Predicted propensity scores of shape (n, a).
    cross_est : bool
        Whether to use cross-estimation.
    
    thres_min : float
        The minimum threshold for the treatment effect.
    thres_diff : float
        The minimum threshold for the difference in treatment effect.
    
    fdx : bool
        Whether to use FDX control, P(FDP > c) < alpha.
    fdx_alpha : float
        The significance level for FDX control.
    fdx_c : float
        The augmentation parameter for FDX control.
    
    verbose : bool
        Whether to print the model information.
    kwargs : dict
        Additional arguments to pass to fit_glm.
    
    Returns
    -------
    df_res : DataFrame
        Dataframe of test results.
    '''

    def estimand(etas, **kwargs):
        eta_0, eta_1 = etas[..., 0], etas[..., 1]
        tau_0, tau_1 = np.mean(eta_0, axis=0), np.mean(eta_1, axis=0)

        tau_1 = np.clip(tau_1, thres_diff, None)
        tau_0 = np.clip(tau_0, thres_diff, None)
        tau_est = np.log(tau_1/tau_0)
        eta_est = eta_1 / tau_1[None,:] -  eta_0 / tau_0[None,:]
        var_est = np.var(eta_est, axis=0, ddof=1)

        # filter out low-expressed genes
        idx = (np.maximum(tau_0,tau_1)<thres_min) & ((tau_1-tau_0)<thres_diff)
        tau_est[idx] = 0.; eta_est[:,idx] = 0.; var_est[idx] = np.inf

        return eta_est, tau_est, var_est

    return compute_causal_estimand(
        estimand, Y, W, A, W_A, family, offset,    
        Y_hat=Y_hat, pi_hat=pi_hat,
        fdx=fdx, fdx_alpha=fdx_alpha, fdx_c=fdx_c, verbose=verbose, **kwargs)





def VIM(eta_est, X, id_covs, **kwargs):
    '''
    Estimate the variable importance measure (VIM) using AIPW.

    Parameters
    ----------
    eta_est : array
        n x p matrix of influence function values.
    '''
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