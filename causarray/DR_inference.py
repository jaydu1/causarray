import numpy as np
import scipy as sp
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests, fdrcorrection


def multiplier_bootstrap(resid, var_est, B):
    '''
    Multiplier bootstrap for inference.

    Parameters
    ----------
    resid : array-like
        [n, p] Residuals.
    var_est : array-like
        [p,] Variance of the parameter estimates.
    B : int
        Number of bootstrap samples.

    Returns
    -------
    z_init : array-like
        [B, p] Bootstrap statistics for each hypothesis.
    '''
    n, p = resid.shape
    B = int(B)
    z_init = np.zeros((B,p))
    eta = resid / np.sqrt(n * var_est[None,:])
    for b in range(B):
        g = np.random.normal(size=n)
        z_init[b, :] = np.sum(eta * g[:, None], axis=0)

    # g = np.random.normal(size=(B,n,1))
    # z_init = np.sum(resid[None, :,:] * g, axis=1) / np.sqrt(n * var_est[None,:])

    return z_init
    

def step_down(tvalues_init, z_init, alpha):
    '''
    Step-down procedure for controlling FWER.

    Parameters
    ----------
    tvalues_init : array-like
        [p,] t-values for each hypothesis.
    z_init : array-like
        [B, p] Bootstrap statistics for each hypothesis.
    alpha : float
        The significance level.

    Returns
    -------
    V : array-like
        [p,] Set of discoveries.
    tvalues : array-like
        [p,] t-values for each hypothesis.
    z : array-like
        [B, p] Bootstrap statistics for each hypothesis.
    '''
    p = z_init.shape[1]
    V = np.zeros(p,)
    z = z_init.copy()
    tvalues = tvalues_init.copy()
    while True:
        tvalues_max = np.max(np.abs(tvalues))
        index_temp = np.unravel_index(np.argmax(np.abs(tvalues)), tvalues.shape)
        z_max = np.max(np.abs(z), axis=1)
        z_max_quan = np.quantile(z_max, 1 - alpha, method='lower')
        if tvalues_max < z_max_quan or z_max_quan == 0:
            break
        tvalues[index_temp] = 0
        z[:,index_temp] = 0
        V[index_temp] = 1
    return V, tvalues, z


def augmentation(V, tvalues, c):
    '''
    Augment the set of discoveries V.

    Parameters
    ----------
    V : array-like
        Set of discoveries.
    tvalues : array-like
        t-values for each hypothesis.
    c : float
        The exceeding level for FDP exceedance rate.

    Returns
    -------
    V : array-like
        Set of discoveries.
    '''
    if c>0:
        size = np.sum(V)
        num_add = int(np.floor(c * size / (1 - c)))
        if num_add >= 1:
            tvalues_sorted = np.sort(np.abs(tvalues), axis=None)[::-1]
            for i in np.arange(num_add):
                V[np.abs(tvalues)==tvalues_sorted[i]] = 1

    return V


def fdx_control(
    tau_est, var_est, tvalues_init, eta_est,
    fdx, B, alpha, c, 
    min_var=1e-4, min_diff=0.1
    ):
    '''
    Perform FDX control.

    Parameters
    ----------
    tau_est : array-like
        The estimated causal effect.    
    var_est : array-like
        The estimated variance.
    tvalues_init : array-like
        The estimated t-values.
    eta_est : array-like
        The estimated influence function values.
    fdx : bool
        If True, perform FDX control.
    B : int
        Number of bootstrap samples.
    alpha : float
        The significance level.
    c : float
        Augmentation parameter.  
    

    Returns
    -------
    V : array-like
        FDX control results, 0 for non-discoveries and 1 for discoveries.
    '''
    n = eta_est.shape[0]

    if fdx:
        id_test = var_est >= min_var
        z_init = multiplier_bootstrap(eta_est, var_est, B)
        z_init[:, ~id_test] = 0.
        tvalues = tvalues_init.copy()
        tvalues[~id_test] = 0.
        V, tvalues, z = step_down(tvalues, z_init, alpha)

        V[(~id_test) & (np.abs(tau_est) > min_diff)] = 1.
        V = augmentation(V, tvalues, c)
    else:
        V = np.zeros(tvalues_init.shape[0])

    return V


def bh_correction(tvalues_init):
    '''
    Perform BH correction.

    Parameters
    ----------
    tvalues_init : array-like
        Initial t-values.

    Returns
    -------
    pvals : array-like
        P-values.
    qvals : array-like
        Q-values.
    pvals_adj : array-like
        P-values after empirical null adjustment.
    qvals_adj : array-like
        Q-values after empirical null adjustment.
    '''
    idx = ~np.isnan(tvalues_init)

    # BH correction
    pvals = np.full(tvalues_init.shape, np.nan)
    qvals = np.full(tvalues_init.shape, np.nan)
    pvals_adj = np.full(tvalues_init.shape, np.nan)
    qvals_adj = np.full(tvalues_init.shape, np.nan)

    if np.sum(idx) > 0:
        pvals[idx] = sp.stats.norm.sf(np.abs(tvalues_init[idx]))*2
        qvals[idx] = multipletests(pvals[idx], alpha=0.05, method='fdr_bh')[1]

        # BH correction with empirical null adjustment
        med = np.nanmedian(tvalues_init[idx])
        mad = sp.stats.median_abs_deviation(tvalues_init[idx], scale="normal", nan_policy='omit')
        
        if mad>0:
            tvalues_init_adj = (tvalues_init - med) / mad

            pvals_adj[idx] = sp.stats.norm.sf(np.abs(tvalues_init_adj[idx]))*2
            qvals_adj[idx] = multipletests(pvals_adj[idx], alpha=0.05, method='fdr_bh')[1]
    
    return pvals, qvals, pvals_adj, qvals_adj