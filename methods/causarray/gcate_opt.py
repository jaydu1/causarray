from causarray.gcate_likelihood import *
from causarray.utils import *
from causarray.gcate_glm import *
import scipy as sp
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict



@njit
def project_l2_ball(X, radius):
    """
    Projects a vector X to the l2 norm ball of a given radius.

    Parameters:
        X (ndarray): The vector to be projected.
        radius (float): The radius of the l2 norm ball.

    Returns:
        ndarray: The projected matrix.
    """
    norms = np.linalg.norm(X)
    if norms > radius:
        X *= radius / norms

    return X



@njit
def line_search(Y, A, x0, g, d, family, nuisance, intercept,
                lam=0., alpha=1., beta=0.5, max_iters=100, tol=1e-3, 
                ):
    """
    Performs line search to find the step size that minimizes a given function.
    
    Parameters:
        f (callable): A scalar-valued function of a vector argument.
        x0 (ndarray): The starting point for the line search.
        g (ndarray): The search direction vector.
        alpha (float, optional): The initial step size. Default is 10.
        beta (float, optional): The shrinkage factor used to reduce the step size. Default is 0.5.
        max_iters (int, optional): The maximum number of iterations. Default is 100.
        tol (float, optional): The tolerance for the step size. Default is 1e-3.
    
    Returns:
        float: The step size that minimizes the function along the search direction.
    """

    # Evaluate the function at the starting point.
    f0 = nll(Y, A, x0, family, nuisance)
    if lam>0.:
        f0 += lam * np.sum(np.abs(x0[intercept:d]))
    
    # Initialize the step size.    
    alpha = type_f(alpha)
    beta = type_f(beta)
    t = alpha
    norm_g = np.linalg.norm(g)

    # Iterate until the maximum number of iterations is reached or the step size is small enough.
    for i in range(max_iters):
        
        # Compute the new point.
        x1 = x0 - t*g
        if lam>0.:
            x1[intercept:d] = np.sign(x1[intercept:d]) * np.maximum(np.abs(x1[intercept:d]) - lam, type_f(0.))
        
        # Evaluate the function at the new point.
        f1 = nll(Y, A, x1, family, nuisance) 
        if lam>0.:
            f1 += lam * np.sum(np.abs(x1[intercept:d]))
        if i==0:
            f01 = f1

        # Check if the function has decreased sufficiently.
        if f1 < f0 - tol*t*norm_g:
            return x1
        if t<1e-4:
            break
        t *= beta

    # Return the maximum step size.
    if f01<2*f1:
        x1 = x0 - alpha*g
        if lam>0.:
            x1[intercept:d] = np.sign(x1[intercept:d]) * np.maximum(np.abs(x1[intercept:d]) - lam, type_f(0.))
    return x1






@njit(parallel=True)
def update(Y, A, B, d, lam, P1, P2,
          family, nuisance, C,
          alpha, beta, max_iters, tol, offset, intercept, num_d):
    n, p = Y.shape
    
    g = grad(Y.T, B, A, family, nuisance.T)
    g[:, :d] = 0.
    for i in prange(n):
        g[i, d:] = project_l2_ball(g[i, d:], 2*C)
        A[i, :] = line_search(Y[i, :].T, B, A[i, :], g[i, :], d, 
                              family, nuisance[0], intercept,
                          type_f(0.), alpha, beta, max_iters, tol)
    if P1 is not None:
        A[:, d:] = P1 @ A[:, d:]
    
    
    g = grad(Y, A, B, family, nuisance)
    if P1 is not None:
        g[:, :d] = 0.
    elif P2 is not None:
        g[:, d-num_d:d] = P2 @ g[:, d-num_d:d]
        g[:, d:] = 0.
    

    for j in prange(p):
        if P2 is None:
            g[j, :] = project_l2_ball(g[j, :], 2*C)
        else:
            g[j, :d] = project_l2_ball(g[j, :d], 2*C)

        B[j, :] = line_search(Y[:, j], A, B[j, :], g[j, :], d, 
                              family, nuisance[:,j], d-num_d,
                              lam, alpha, beta, max_iters, tol
                             )

    if P2 is None:
        B[:, d:] = np.clip(B[:, d:], -10., 10.)
    func_val = nll(Y, A, B, family, nuisance)

    return func_val, A, B


def alter_min(
    Y, r, X=None, P1=None, P2=None, 
    A=None, B=None,
    kwargs_glm={}, kwargs_ls={}, kwargs_es={}, 
    lam=0., num_d=None, intercept=1, offset=1, verbose=False, update_disp=False, C=1e5):
    '''
    Alternative minimization of latent factorization for generalized linear models.

    Parameters:
        Y (ndarray): The response matrix.        
        r (int): The rank of the factorization.
        X (ndarray): The matrix to be projected.
        P1 (ndarray): The projection matrix for the rows.
        P2 (ndarray): The projection matrix onto the orthogonal column space of Gamma.        
        A (ndarray): The initial matrix for the covariate and latent factors. If None, initialize internally.
        B (ndarray): The initial matrix for the covariate and latent coefficients. If None, initialize internally.
        kwargs_glm (dict): Keyword arguments for the generalized linear model.
        kwargs_ls (dict): Keyword arguments for the line search.
        kwargs_es (dict): Keyword arguments for the early stopping.
        lam (float): The regularization parameter for the l1 norm of the coefficients.
        num_d (int): The number of columns to be regularized. Assume the last 'num_d' columns of the covariates are the regularized coefficients. If 'num_d' is None, it is set to be 'd-offset-intercept' by default.
        C (float): The maximum radius of the l2 norm of the gradient.
        intercept (int): The indicator of whether to include the intercept term in the covariates.
        offset (int): The indicator of whether to include the offset term in the covariates.
        verbose (bool): The indicator of whether to print the progress.

    Returns:
        A (ndarray): The matrix for the covariate and updated latent factors.
        B (ndarray): The matrix for the updated covariate and latent coefficients.
        info (dict): A dictionary containing the information of the optimization.
    '''
    n, p = Y.shape
    d = X.shape[1]
    
    assert d>0
    if verbose:
        pprint.pprint({'n':n,'p':p,'d':d,'r':r})

    kwargs_glm = {**{'family':'gaussian', 'nuisance':np.ones(p)}, **kwargs_glm}
    kwargs_ls = {**{'alpha':0.1, 'beta':0.5, 'max_iters':20, 'tol':1e-4}, **kwargs_ls}
    kwargs_es = {**{'max_iters':500, 'warmup':5, 'patience':20, 'tolerance':1e-4}, **kwargs_es}
    
    
    if verbose:
        pprint.pprint({'kwargs_glm':kwargs_glm,'kwargs_ls':kwargs_ls,'kwargs_es':kwargs_es})
    
    family, nuisance = kwargs_glm['family'], kwargs_glm['nuisance'].astype(type_f)
    nuisance = nuisance.reshape(1,-1)

    # if C is None:
    #     C = 1e3 if family=='nb' else 1e5

    if P1 is True:
        Q, _ = sp.linalg.qr(X[:,offset:], mode='economic')
        P1 = np.identity(n) - Q @ Q.T
        P1 = P1.astype(type_f)
            
    if num_d is None:
        num_d = d-offset-intercept
        
    

    # initialization for Theta = A @ B^T
    if A is None or B is None:
        A = np.empty((n, r), dtype=type_f)
        B = np.empty((p, d+r), dtype=type_f)
        E = init_inv_link(Y, family, nuisance)
        
        if d>0:
            A = np.c_[X, A]
            
            offset_arr = None if offset==0 else X[:,0]
            alpha = np.full(d-offset, 1e-8)
            alpha[:intercept] = 0.
            B[:, offset:d] = fit_glm(Y, X[:,offset:],
                offset=offset_arr, family=family, disp_glm=nuisance[0])[0]
            
            E = P1 @ E
        u, s, vh = sp.sparse.linalg.svds(E, k=r)        
        A[:, d:] = u * s[None,:]**(1/2)
        B[:, d:] = vh.T * s[None,:]**(1/2)
        del E, u, s, vh

        if offset==1:
            scale = np.sqrt(np.median(np.abs(X[:,0])))
            B[:, :offset] = scale
            A[:, :offset] /= scale 

    if P2 is not None:  
        P2 = P2.astype(type_f)
        E = A[:,d-num_d:] @ B[:,d-num_d:].T @ (np.identity(p) - P2)
        u, s, vh = sp.sparse.linalg.svds(E, k=r)
        B[:, d-num_d:d] = P2 @ B[:, d-num_d:d]
        A[:, d:] = u * s[None,:]**(1/2)
        B[:, d:] = vh.T * s[None,:]**(1/2)
        del E, u, s, vh

    
                
    Y = Y.astype(type_f)
    A = A.astype(type_f)
    B = B.astype(type_f)
    lam = type_f(lam)
    
    assert ~np.any(np.isnan(A))
    assert ~np.any(np.isnan(B))
    
    func_val_pre = nll(Y, A, B, family, nuisance)/p + lam * np.mean(np.abs(B[:,d-num_d:d]))
    hist = [func_val_pre]
    es = Early_Stopping(**kwargs_es)
    with tqdm(np.arange(kwargs_es['max_iters'])) as pbar:
        for t in pbar:
            func_val, A, B = update(
                Y, A, B, d, lam, P1, P2,
                family, nuisance, C,
                kwargs_ls['alpha'], kwargs_ls['beta'], kwargs_ls['max_iters'], kwargs_ls['tol'], 
                offset,intercept,num_d
            )
            func_val = func_val/p + lam * np.mean(np.abs(B[:,d-num_d:d]))
            hist.append(func_val)
            if not np.isfinite(func_val) or func_val>np.maximum(1e3*np.abs(func_val_pre),1e3):
                print('Encountered large or infinity values. Try to decrease the value of C for the norm constraints.')
                break
            elif es(func_val):
                pbar.set_postfix_str('Early stopped.' + es.info)
                pbar.close()                
                break
            else:
                func_val_pre = func_val
            pbar.set_postfix(nll='{:.02f}'.format(func_val))

            if update_disp and family=='nb' and t > es.warmup:
                nuisance = estimate_disp(Y, Y_hat=np.exp(A @ B.T)).reshape(1,-1)

    kwargs_glm['nuisance'] = nuisance[0]
    info = {'n_iter':t, 'func_val':func_val, 'resid':func_val_pre - func_val,
           'hist':hist, 'kwargs_glm':kwargs_glm}
    return A, B, info



