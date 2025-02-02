from causarray.gcate_opt import *
import pandas as pd
from causarray.utils import comp_size_factor, _filter_params


def _check_input(Y, X, family, disp_glm, disp_family, offset, c1, **kwargs):
    if not (X.ndim == 2 and Y.ndim == 2):
        raise ValueError("Input must have ndim of 2. Y.ndim: {}, X.ndim: {}.".format(Y.ndim, X.ndim))

    if not np.allclose(Y, Y.astype(int)):
        warnings.warn("Y is not integer-valued. It will be rounded to the nearest integer.")
        Y = np.round(Y)

    if np.sum(np.any(Y!=0., axis=0))<Y.shape[1]:
        raise ValueError("Y contains non-expressed features.")

    if np.linalg.svd(X, compute_uv=False)[-1] < 1e-3:
        raise ValueError("The covariate matrix is near singular.")
    
    Y = Y.astype(type_f)
    n, p = Y.shape

    kwargs_glm = {}
    kwargs_glm['family'] = family

    if offset is not None and offset is not False:
        if type(offset)==bool and offset is True:
            size_factor = comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs))
        kwargs_glm['size_factor'] = size_factor 
        offset = np.log(size_factor)
    else:
        offset = None
    if kwargs_glm['family']=='nb':
        if disp_family is None:
            disp_family = 'poisson'
        disp_glm = estimate_disp(Y, X, offset=offset, disp_family=disp_family, maxiter=1000)
    if disp_glm is not None:
        kwargs_glm['nuisance'] = disp_glm
            
    kwargs_glm = {**{'family':'gaussian', 'nuisance':np.ones((1,p)), 'size_factor':np.ones((n,1))
    }, **kwargs_glm}

    c1 = 0.05 if c1 is None else c1
    lam1 = c1 #* num_d #* np.sqrt(np.log(p)/n)

    return Y, kwargs_glm, lam1


def fit_gcate(Y, X, r, family='nb', disp_glm=None, disp_family=None, offset=True,
    kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    c1=None, num_d=1, verbose=False, **kwargs
):
    '''
    Parameters
    ----------
    Y : array-like, shape (n, p)
        The response variable.
    X : array-like, shape (n, d)
        The covariate matrix.
    r : int
        The number of unmeasured confounders.
    family : str
        The family of the GLM. Default is 'poisson'.
    disp_glm : array-like, shape (p, ) or None
        The dispersion parameter for the negative binomial distribution.
    offset : array-like, shape (p, ) or None
        The offset parameter.
    kwargs_ls_1 : dict
        Keyword arguments for the line search solver in the first phrase.
    kwargs_ls_2 : dict
        Keyword arguments for the line search solver in the second phrase.
    kwargs_es_1 : dict
        Keyword arguments for the early stopper in the first phrase.
    kwargs_es_2 : dict
        Keyword arguments for the early stopper in the second phrase.
    c1 : float
        The regularization constant in the first phrase. Default is 0.1.
    num_d : int
        The number of covariates to be regularized. Assume the last num_d covariates are to be regularized. Default is 1.
    verbose : bool
        Print the optimization information.
    **kwargs : dict
        Additional keyword arguments.
    '''
    Y, kwargs_glm, lam1 = _check_input(Y, X, family, disp_glm, disp_family, offset, c1, **kwargs)    

    r = int(r)

    A01, A02, P_Gamma, A1, A2, info = estimate(Y, X, r, num_d, 
        lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, verbose, **kwargs)
        
    return A1, A2, info, A01, A02


def estimate(Y, X, r, num_d, lam1, 
    kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, verbose=False, **kwargs):
    '''
    Two-stage estimation of the GCATE model.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    X : array-like, shape (n, d)
        Observed covariate matrix.
    r : int
        Number of latent variables.
    num_d : int
        The number of columns to be regularized. Assume the last 'num_d' columns of the covariates are the regularized coefficients. If 'num_d' is None, it is set to be 'd' by default.
    lam1 : float
        Regularization parameter for the first optimization problem.
    kwargs_glm : dict
        Keyword arguments for the GLM.
    kwargs_ls_1 : dict
        Keyword arguments of the line search algorithm for the first optimization problem.
    kwargs_ls_2 : dict
        Keyword arguments of the line search algorithm for the second optimization problem.
    kwargs_es_1 : dict
        Keyword arguments of the early stopping monitor for the first optimization problem.
    kwargs_es_2 : dict
        Keyword arguments of the early stopping monitor for the second optimization problem.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    A01 : array-like, shape (n, d+r)
        The observed covaraite and unobserved uncorrelated latent factors.
    A02 : array-like, shape (p, d+r)  
        The estimated marginal effects and latent coefficients.
    P_Gamma : array-like, shape (p, p)
        The projection matrix for the second optimization problem.
    A1 : array-like, shape (n, d+r)
        The observed covaraite and unobserved latent factors.
    A2 : array-like, shape (p, d+r)
        The estimated primary effects and latent coefficients.
    '''
    d = X.shape[1]
    p = Y.shape[1]

    valid_params = _filter_params(alter_min, kwargs)

    A01, A02, info = alter_min(
        Y, r, X=X, P1=True,
        kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_1, kwargs_es=kwargs_es_1, verbose=verbose, **valid_params)
    Q, _ = sp.linalg.qr(A02[:,d:], mode='economic')
    P_Gamma = np.identity(p) - Q @ Q.T    

    if lam1 == 0.:
        A1, A2 = A01, A02
    else:
        A1, A2, info = alter_min(
            Y, r, X=X, P2=P_Gamma, A=A01.copy(), B=A02.copy(), lam=lam1, num_d=num_d,
            kwargs_glm=info['kwargs_glm'], kwargs_ls=kwargs_ls_2, kwargs_es=kwargs_es_2, verbose=verbose, **valid_params)
    return A01, A02, P_Gamma, A1, A2, info


def estimate_r(Y, X, r_max, c=1., 
    family='nb', disp_glm=None, disp_family='poisson', offset=True,
    kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    **kwargs
):
    '''
    Estimate the number of latent factors for the GCATE model.

    Parameters
    ----------
    Y : array-like, shape (n, p)
        Response matrix.
    X : array-like, shape (n, d)
        Observed covariate matrix.
    r_max : int
        Number of latent variables.
    c : float
        The constant factor for the complexity term.
    family : str
        The family of the GLM. Default is 'poisson'.
    disp_glm : array-like, shape (1, p) or None
        The dispersion parameter for the negative binomial distribution.
    kwargs_glm : dict
        Keyword arguments for the GLM.
    kwargs_ls_1 : dict
        Keyword arguments of the line search algorithm for the first optimization problem.
    kwargs_ls_2 : dict
        Keyword arguments of the line search algorithm for the second optimization problem.
    kwargs_es_1 : dict
        Keyword arguments of the early stopping monitor for the first optimization problem.
    kwargs_es_2 : dict
        Keyword arguments of the early stopping monitor for the second optimization problem.

    Returns
    -------
    df_r : DataFrame
        Results of the number of latent factors.
    '''
    Y, kwargs_glm, _ = _check_input(Y, X, family, disp_glm, disp_family, offset, None, **kwargs)
    
    family, nuisance, size_factor = kwargs_glm['family'], kwargs_glm['nuisance'], kwargs_glm['size_factor']
    nuisance = nuisance.reshape(1,-1)
    size_factor = size_factor.reshape(-1,1)
    
    d = X.shape[1]
    n, p = Y.shape

    res = []
    if np.isscalar(r_max):
        r_list = np.arange(1, int(r_max)+1)
    else:
        r_list = np.array(r_max, dtype=int)
        
    for r in r_list:
        A01, A02, _, A1, A2, _ = estimate(Y, X, r, 1,
            0, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, **kwargs)

        logh = log_h(Y, family, nuisance)

        if r==1:
            ll = 2 * ( 
                nll(Y, A01, A02, family, nuisance, size_factor) / p 
                - np.sum(logh) / (n*p) ) 
            nu = d * np.maximum(n,p) * np.log(n * p / np.maximum(n,p)) / (n*p)
            jic = ll + c * nu
            res.append([0, ll, nu, jic])
        
        ll = 2 * ( 
            nll(Y, A1, A2, family, nuisance, size_factor) / p 
            - np.sum(logh) / (n*p) ) 
        nu = (d + r) * np.maximum(n,p) * np.log(n * p / np.maximum(n,p)) / (n*p)
        jic = ll + c * nu
        res.append([r, ll, nu, jic])

    df_r = pd.DataFrame(res, columns=['r', 'deviance', 'nu', 'JIC'])
    return df_r 