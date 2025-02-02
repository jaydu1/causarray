import numpy as np
# from scipy.special import expit, xlogy, xlog1py, logsumexp, factorial
# from scipy.stats import binom, poisson, norm, nbinom

import numba as nb
from numba import njit, prange

type_f = np.float64


from scipy.special import xlogy, gammaln
from numba.extending import get_cython_function_address
import ctypes

_PTR = ctypes.POINTER
_dble = ctypes.c_double
_ptr_dble = _PTR(_dble)

addr = get_cython_function_address("scipy.special.cython_special", "gammaln")
functype = ctypes.CFUNCTYPE(_dble, _dble)
gammaln_float64 = functype(addr)

@nb.vectorize
def gammaln_nb(x):
  return gammaln_float64(x)


def log_h(y, family, nuisance):
    if family=='nb':
        return gammaln(y + nuisance) - gammaln(nuisance) - gammaln(y+1)
    elif family=='poisson':
        return - gammaln(y+1)


@nb.vectorize
def log1mexp(a):
    '''
    A numeral stable function to compute log(1-exp(a)) for a in [-inf,0].
    '''
    if(a >= -np.log(type_f(2.))):
        return np.log(-np.expm1(a)) 
    else:
        return np.log1p(-np.exp(a))
    

@njit
def nll(Y, A, B, family, nuisance=np.ones((1,1)), Tys=np.zeros((1,1)), thres_disp=10.
    #  size_factor=np.ones((1,1))
    ):
    """
    Compute the negative log likelihood for generalized linear models with optional nuisance parameters.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    A : array-like of shape (n_samples, n_factors)
        The input data matrix.
    B : array-like of shape (n_features, n_factors)
        The input data matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'poisson', and 'nb'.
    nuisance : float or array-like of shape (n_samples,), optional (default=1)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter.
    size_factor : float or array-like of shape (n_samples,), optional (default=1)
        The size factor for the response variable.
    
    Returns:
    nll : float
        The negative log likelihood.
    """
    
    Theta = A @ B.T
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'poisson':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b = np.exp(Theta)
    elif family == 'nb':
        Xi = np.clip(Theta, -np.inf, type_f(1e2))
        exp_Xi = np.exp(Xi)
        tmp = np.clip(1 / (type_f(1.) + exp_Xi / nuisance), 1e-6, 1-1e-6)

        Theta = np.where(nuisance > thres_disp, Xi, np.log1p(-tmp))
        b = np.where(nuisance > thres_disp, exp_Xi, - nuisance * np.log(tmp) #+ gammaln_nb(nuisance+Y) - gammaln_nb(nuisance)
        ) # ignoring a common factor - gammaln_nb(Y)
    else:
        raise ValueError('Family not recognized')

    nll = - np.sum(Ty * Theta - b + Tys) / type_f(n) # * size_factor to get back the likelihood

    return nll



@njit
def grad(Y, A, B, family, nuisance=np.ones((1,1)), thres_disp=10.
    #size_factor=np.ones((1,1)),
        ):
    """
    Compute the gradient of log likelihood with respect to B
    for generalized linear models with optional nuisance parameters.
    
    The natural parameter of Y is Theta = A @ B^T.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    A : array-like of shape (n_samples, n_factors)
        The input data matrix.
    B : array-like of shape (n_features, n_factors)
        The input data matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'poisson', and 'nb'.
    nuisance : float or array-like of shape (n_samples,), optional (default=1)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter.
    
    Returns:
    grad : array-like of shape (n_features, n_factors)
        The gradient of log likelihood.
    """
    Theta = A @ B.T
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'nb':
        Xi = np.clip(Theta, -np.inf, type_f(1e2))
        b_p = np.exp(Xi)
        tmp = np.clip(1 / (type_f(1.) + b_p / nuisance), 1e-6, 1-1e-6)
        grad = - (Ty - b_p) * np.where(nuisance > thres_disp, 1., tmp) # * size_factor to get back the likelihood
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b_p = np.exp(Theta)
        grad = - (Ty - b_p) # * size_factor to get back the likelihood
    else:
        raise ValueError('Family not recognized')
    grad = grad.T @ A / type_f(n)
    return grad

