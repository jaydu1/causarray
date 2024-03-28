from causarray.utils import *
import numpy as np
import statsmodels as stats
import statsmodels.api as sm
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')


def init_inv_link(Y, family, disp):
    if family=='gaussian':
        val = Y/disp
    elif family=='poisson':
        val = np.log1p(Y)
    elif family=='nb':
        val = np.log1p(Y)
    elif family=='binomial':
        eps = (np.mean(Y, axis=0) + np.mean(Y, axis=1)) / 2 
        val = np.log((Y + eps)/(disp - Y + eps))
    else:
        raise ValueError('Family not recognized')
    return val



def fit_glm(Y, X, A=None, family='gaussian', 
    disp_glm=None, impute=False, offset=False, 
    alpha=0., method='glm', n_jobs=-3):
    '''
    Fit GLM to each column of Y, with covariate X and treatment A.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes
    X : array
        n x d matrix of covariates
    A : array
        n x 1 vector of treatments or None
    family : str
        family of GLM to fit, can be one of: 'gaussian', 'poisson', 'nb'
    disp_glm : array or None
        dispersion parameter for negative binomial GLM
    return_df : bool
        whether to return results as DataFrame
    impute : bool
        whether to impute potential outcomes and get predicted values
    offset : bool
        whether to use log of sum of Y as offset    
    '''
    
   
    if A is None:
        assert impute is False
    else:
        X = np.c_[X,A]

    # if impute is not False:
    #     Yhat_0 = []
    #     Yhat_1 = []
    # else:
    #     Yhat = []

    if offset:
        offsets = np.log(np.sum(Y, axis=1))
    else:
        offsets = None
    
    if method=='mle':
        funcs = {
            'gaussian': sm.GLM,
            'poisson': stats.discrete.discrete_model.Poisson,
            'zip': stats.discrete.count_model.ZeroInflatedPoisson,
            'nb': stats.discrete.discrete_model.NegativeBinomial,
            'zinb':stats.discrete.count_model.ZeroInflatedNegativeBinomialP
        }
        func = funcs.get(family, lambda: ValueError('family must be one of: "gaussian", "poisson", "nb"'))

    elif method=='glm':
        # estimate dispersion parameter for negative binomial GLM if not provided
        if family=='nb' and disp_glm is None:
            disp_glm = estimate_disp(Y, X, A)       

        func = sm.GLM
        families = {
            'gaussian': lambda disp: sm.families.Gaussian(),
            'poisson': lambda disp: sm.families.Poisson(),
            'nb': lambda disp: sm.families.NegativeBinomial(alpha=1/disp)
        }

    d = X.shape[1]
    if impute is not False and isinstance(impute, np.ndarray):
        X_test = impute
    else:
        X_test = X[:,:-1]
            
    # B = []
    # disp = []
    
    # for j in range(Y.shape[1]):
    #     # glm_family = families.get(family, lambda: ValueError('family must be one of: "gaussian", "poisson", "nb"'))(disp_glm[j] if family == 'nb' else None)
        
    #     try:
    #         # mod = sm.GLM(Y[:,j], X, family=glm_family, offset=offsets).fit()
    #         mod = func(Y[:,j], X, offset=offsets).fit(disp=False)
    #         B.append(mod.params[:d])            
    #         append_values(impute, Y, j, X, A, offsets, mod)

    #         alpha = 1. if len(mod.params)==d else mod.params[-1]            
    #         disp.append(1./ alpha)
    #     except:
    #         B.append(np.full(d, np.nan))
    #         append_values(impute, Y, j, X, A, offsets)
    #         disp.append(1.)
    # B = np.array(B)
    # disp = np.array(disp)


    

    def fit_model(j, Y, X, func, offsets, family, disp, d, impute, alpha, method=method):        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")

                if method=='mle':
                    if alpha==0.:
                        mod = func(Y[:,j], X, offset=offsets).fit(disp=False)
                        if not mod.converged or mod.params[-1]<1e-3 or mod.params[-1]>1e3:
                            mod = stats.discrete.discrete_model.NegativeBinomial(Y[:,j], X).fit_regularized(
                                method='l1', alpha=1e-1, qc_verbose=False, disp=False)
                    else:
                        mod = func(Y[:,j], X, offset=offsets).fit_regularized(alpha=alpha, qc_verbose=False, disp=False)
                    
                elif method=='glm':
                    glm_family = families.get(family, lambda: ValueError('family must be one of: "gaussian", "poisson", "nb"'))(disp_glm[j] if family == 'nb' else None)
                    mod = func(Y[:,j], X, family=glm_family, offset=offsets).fit_regularized(alpha=alpha)       
                
            B = mod.params[:d]

            if impute is not False:
                Yhat_0 = mod.predict(np.c_[X_test, np.zeros((X_test.shape[0],1))], offset=offsets)
                Yhat_1 = mod.predict(np.c_[X_test, np.ones((X_test.shape[0],1))], offset=offsets)
            else:
                Yhat_0 = Yhat_1 = mod.predict(X, offset=offsets)

            inv_theta = 1. if len(mod.params)==d else mod.params[-1]            
            disp = 1./ np.clip(inv_theta, 0.01, 100.)
        except:
            B = np.full(d, np.nan)
            Yhat_0 = Yhat_1 = np.full(X_test.shape[0], np.mean(Y[:,j]))
            disp = 1.
        return B, disp, Yhat_0, Yhat_1


    # results = []
    # for j in range(Y.shape[1]):
    #     results.append(fit_model(j, Y, X, func, offsets, family, disp_glm, d, impute, alpha))
    results = Parallel(n_jobs=n_jobs)(delayed(fit_model)(
        j, Y, X, func, offsets, family, disp_glm, d, impute, alpha) for j in range(Y.shape[1]))

    B, disp, Yhat_0, Yhat_1 = zip(*results)
    B = np.array(B)
    disp = np.array(disp)
    Yhat_0 = np.array(Yhat_0).T
    Yhat_1 = np.array(Yhat_1).T

    if impute is not False:        
        Yhat = (Yhat_0, Yhat_1)
    else:
        Yhat = np.array(Yhat_0)
    
    return B, Yhat, disp    


def estimate_disp(Y, X, A=None, method='mom', **kwargs):
    if method=='mom':
        _, Y_hat, _ = fit_glm(Y, X, A, family='poisson', impute=False, **kwargs)
        # mu_glm = np.mean(Y_hat, axis=0)
        # disp_glm = (np.mean((Y - mu_glm[None,:])**2, axis=0) - mu_glm) / mu_glm**2
        disp_glm = np.mean((Y - Y_hat)**2 - Y_hat, axis=0) / np.mean(Y_hat**2, axis=0)
        # disp_glm = np.mean(((Y - Y_hat)**2 - Y_hat) / Y_hat**2, axis=0)
        disp_glm = 1./np.clip(disp_glm, 0.01, 100.)
    elif method=='mle':
        _, _, disp_glm = fit_glm(Y, X, A, family='nb', impute=False, **kwargs)

    return disp_glm