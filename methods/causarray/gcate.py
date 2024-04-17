from causarray.gcate_opt import *
import pandas as pd


def fit_gcate(Y, X, r, family='poisson', disp_glm=None, offset=None,
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
    C : float
        The constant for maximum l2 norm of the coefficients. Default is 1e5.
    '''
    if not (X.ndim == 2 and Y.ndim == 2):
        raise ValueError("Input must have ndim of 2. Y.ndim: {}, X.ndim: {}.".format(Y.ndim, X.ndim))

    if np.sum(np.any(Y!=0., axis=0))<Y.shape[1]:
        raise ValueError("Y contains non-expressed features.")

    kwargs_glm = {}
    kwargs_glm['family'] = family

    if kwargs_glm['family']=='nb':
        disp_glm = estimate_disp(Y, X)
    if disp_glm is not None:
        kwargs_glm['nuisance'] = disp_glm
    if offset is not None:
        if type(offset)==bool and offset is True:
            offset = comp_size_factor(Y, **kwargs)
        kwargs_glm['offset'] = offset
        
    d = X.shape[1]
    p = Y.shape[1]
    n = Y.shape[0]

    r = int(r)

    c1 = 0.02 if c1 is None else c1
    lam1 = c1 * np.sqrt(np.log(p)/n)

    _, _, P_Gamma, A1, A2, info = estimate(Y, X, r, num_d, 
        lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, verbose, **kwargs)
        
    return A1, A2, info


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

    A01, A02, info = alter_min(
        Y, r, X=X, P1=True,
        kwargs_glm=kwargs_glm, kwargs_ls=kwargs_ls_1, kwargs_es=kwargs_es_1, verbose=verbose, update_disp=True, **kwargs)
    Q, _ = sp.linalg.qr(A02[:,d:], mode='economic')
    P_Gamma = np.identity(p) - Q @ Q.T

    # A1 = A01.copy()
    # A1[:,d:] += A01[:,:d] @ np.linalg.solve(A02[:,d:].T @ A02[:,d:], A02[:,d:].T @ A02[:,:d]).T
    # A2 = A02.copy()
    A1, A2, info = alter_min(
        Y, r, X=X, P2=P_Gamma, A=A01.copy(), B=A02.copy(), lam=lam1, num_d=num_d,
        kwargs_glm=info['kwargs_glm'], kwargs_ls=kwargs_ls_2, kwargs_es=kwargs_es_2, verbose=verbose, **kwargs)
    return A01, A02, P_Gamma, A1, A2, info


def estimate_r(Y, X, r_max, c=1., family='poisson', disp_glm=None, offset=None,
    kwargs_ls_1={}, kwargs_ls_2={}, kwargs_es_1={}, kwargs_es_2={},
    c1=None, num_d=1, **kwargs
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
    c1 : float
        Regularization parameter for the second optimization problem.
    num_d : int
        Number of latent variables.

    Returns
    -------
    df_r : DataFrame
        Results of the number of latent factors.
    '''
    if not (X.ndim == 2 and Y.ndim == 2):
        raise ValueError("Input must have ndim of 2. Y.ndim: {}, X.ndim: {}.".format(Y.ndim, X.ndim))

    if np.sum(np.any(Y!=0., axis=0))<Y.shape[1]:
        raise ValueError("Y contains non-expressed features.")
    
    d = X.shape[1]
    p = Y.shape[1]
    n = Y.shape[0]
    Y = Y.astype(type_f)

    kwargs_glm = {}
    kwargs_glm['family'] = family

    if kwargs_glm['family']=='nb':
        disp_glm = estimate_disp(Y, X)
    if disp_glm is not None:
        kwargs_glm['nuisance'] = disp_glm
    if offset is not None:
        if type(offset)==bool and offset is True:
            offset = comp_size_factor(Y, **kwargs)
        kwargs_glm['offset'] = offset
            
    kwargs_glm = {**{'family':'gaussian', 'nuisance':np.ones((1,p)), 'offset':np.ones((n,1))}, **kwargs_glm}
    family, nuisance, offset = kwargs_glm['family'], kwargs_glm['nuisance'], kwargs_glm['offset']

    c1 = 0.1 if c1 is None else c1
    lam1 = c1 * np.sqrt(np.log(p)/n)

    res = []
    if np.isscalar(r_max):
        r_list = np.arange(1, int(r_max)+1)
    else:
        r_list = np.array(r_max, dtype=int)
        
    for r in r_list:
        A01, A02, _, A1, A2, _ = estimate(Y, X, r, num_d,
            lam1, kwargs_glm, kwargs_ls_1, kwargs_es_1, kwargs_ls_2, kwargs_es_2, **kwargs)

        logh = log_h(Y, family, nuisance)

        if r==1:
            ll = 2 * ( 
                nll(Y, A01, A02, family, nuisance, offset) / p 
                - np.sum(logh) / (n*p) ) 
            nu = d * np.maximum(n,p) * np.log(n * p / np.maximum(n,p)) / (n*p)
            jic = ll + c * nu
            res.append([0, ll, nu, jic])
        
        ll = 2 * ( 
            nll(Y, A1, A2, family, nuisance, offset) / p 
            - np.sum(logh) / (n*p) ) 
        nu = (d + r) * np.maximum(n,p) * np.log(n * p / np.maximum(n,p)) / (n*p)
        jic = ll + c * nu
        res.append([r, ll, nu, jic])

    df_r = pd.DataFrame(res, columns=['r', 'deviance', 'nu', 'JIC'])
    return df_r    