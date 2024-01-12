import numpy as np
import statsmodels.api as sm
from scipy.stats import norm



def multiplier_bootstrap(resid, theta_var, B):
    '''
    Multiplier bootstrap for inference.

    Parameters
    ----------
    resid : array-like
        [n, p] Residuals.
    theta_var : array-like
        [p,] Variance of the parameter estimates.
    B : int
        Number of bootstrap samples.

    Returns
    -------
    z_init : array-like
        [B, p] Bootstrap statistics for each hypothesis.
    '''
    n, p = resid.shape
    z_init = np.zeros((B,p))
    for b in range(B):
        g = np.random.normal(size=n)
        temp = np.sum(resid * g[:, None], axis=0)
        z_init[b, :] = theta_var**(-0.5) * temp / np.sqrt(n)
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
        z_max_quan = np.quantile(z_max, 1-alpha)
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


def comp_stat(true, pred, c):

    # type I error
    typeI_err = np.sum(true[pred==1]==0)/np.sum(true==0)

    # false discovery proportion
    FDP = np.sum(true[pred==1]==0)/np.sum(pred==1)

    # power
    power = np.sum(true[pred==1]==1)/np.sum(true==1)

    # false discovery rate exceedance
    FDPex = (FDP > c).astype(int)

    # number of discoveries
    num_dis = np.sum(pred==1)

    return [typeI_err, FDP, power, FDPex, num_dis]