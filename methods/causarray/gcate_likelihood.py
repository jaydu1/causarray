import numpy as np
# from scipy.special import expit, xlogy, xlog1py, logsumexp, factorial
# from scipy.stats import binom, poisson, norm, nbinom

import numba as nb
from numba import njit, prange

type_f = np.float64


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
def nll(Y, A, B, family, nuisance=np.ones((1,1))):
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
        The family of the generalized linear model. Options include 'gaussian', 'binomial', 'poisson',
        and 'negative_binomial'.
    nuisance : float or array-like of shape (n_samples,), optional (default=None)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter. If None, the default value of the nuisance parameter is used.
    
    Returns:
    nll : float
        The negative log likelihood.
    """
    
    Theta = A @ B.T
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'binomial':
        b = nuisance * np.log(type_f(1.) + np.exp(Theta))
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b = np.exp(Theta)
    elif family == 'gaussian':
        b = Theta**2/type_f(2.)
        Ty /= np.sqrt(nuisance)
    elif family == 'negative_binomial':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b = - nuisance * np.log1p(np.exp(Theta))
    else:
        raise ValueError('Family not recognized')
    nll = - np.sum(Ty * Theta - b) / type_f(n)
    return nll



@njit
def grad(Y, A, B, family, nuisance=np.ones((1,1)), 
         direct=False):
    """
    Compute the gradient of log likelihood with respect to B
    for generalized linear models with optional nuisance parameters.
    
    The natural parameter of Y is A @ B^T.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    A : array-like of shape (n_samples, n_factors)
        The input data matrix.
    B : array-like of shape (n_features, n_factors)
        The input data matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'gaussian', 'binomial', 'poisson',
        and 'negative_binomial'.
    nuisance : float or array-like of shape (n_samples,), optional (default=None)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter. If None, the default value of the nuisance parameter is used.
    
    Returns:
    grad : array-like of shape (n_features, n_factors)
        The gradient of log likelihood.
    """
    Theta = A @ B.T
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'negative_binomial':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        # grad with respect to theta
        b_p = np.exp(Theta) 
        # grad with respect to xi
        grad = - (Ty - b_p) / (type_f(1.) + np.exp(Theta)/nuisance) 
        
        if not direct:
            grad = grad.T @ A / type_f(n)

        return grad
    
    if family == 'binomial':
        b_p = nuisance / (type_f(1.) + np.exp(-Theta))
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b_p = np.exp(Theta)
    elif family == 'gaussian':
        b_p = Theta
        Ty /= np.sqrt(nuisance)
    else:
        raise ValueError('Family not recognized')
        
    if direct:
        return - (Ty - b_p)
    else:
        grad = - (Ty - b_p).T @ A / type_f(n)
        return grad



@njit
def hess(Y, Theta, family, nuisance=np.ones((1,1))):
    """
    Compute the gradient of log likelihood with respect to B
    for generalized linear models with optional nuisance parameters.
    
    The natural parameter of Y is A @ B^T.
    
    Parameters:
    Y : array-like of shape (n_samples, n_features)
        The response variable.
    Theta : array-like of shape (n_samples, n_features)
        The natural parameter matrix.
    family : str, optional (default='gaussian')
        The family of the generalized linear model. Options include 'gaussian', 'binomial', 'poisson',
        and 'negative_binomial'.
    nuisance : float or array-like of shape (n_samples,), optional (default=None)
        The nuisance parameter for the family. For the Gaussian family, this is the variance; for the Poisson
        family, this is the scaling factor; and for the negative binomial family, this is the overdispersion
        parameter. If None, the default value of the nuisance parameter is used.
    
    Returns:
    hess : float
        The hessian of the log likelihood with respect to Theta.
    """
    Ty = Y.copy()
    n = Y.shape[0]
    
    if family == 'binomial':
        b_pp = nuisance * np.exp(-Theta) / (type_f(1.) + np.exp(-Theta))**2
    elif family == 'poisson':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b_pp = np.exp(Theta)
    elif family == 'gaussian':
        b_pp = np.ones_like(Theta)
    elif family == 'negative_binomial':
        Theta = np.clip(Theta, -np.inf, type_f(1e2))
        b_pp = np.exp(Theta) / (type_f(1.) + np.exp(Theta) / nuisance)
    else:
        raise ValueError('Family not recognized')
    hess = b_pp
    return hess