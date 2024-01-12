import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests



def fit_glm(Y, X, A, family='gaussian', disp_glm=None, return_df=False, impute=False, offset=False):
    '''
    Fit GLM to each column of Y, with covariate X and treatment A.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes
    X : array
        n x d matrix of covariates
    A : array
        n x 1 vector of treatments
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
    
    # estimate dispersion parameter for negative binomial GLM if not provided
    if family=='nb' and disp_glm is None:
        _, Y_hat, _, _, _ = fit_glm(Y, X, A, family='poisson', impute=False)
        mu_glm = np.mean(Y_hat, axis=0)
        disp_glm = (np.mean((Y - mu_glm[None,:])**2, axis=0) - mu_glm) / mu_glm**2
        disp_glm = np.clip(disp_glm, 0.01, 100.)

    X = np.c_[X,A]

    tvals = []
    pvals = []
    resid_response = []
    if impute:
        Yhat_0 = []
        Yhat_1 = []
    else:
        Yhat = []

    if offset:
        offsets = np.log(np.sum(Y, axis=1))
    else:
        offsets = None
    
    def append_values(impute, Y, j, X, A, offsets, mod=None):
        if impute:
            Yhat_0.append(mod.predict(np.c_[X[:,:-1], np.zeros_like(A)], offset=offsets) if mod else np.full(Y.shape[0], np.mean(Y[:,j])))
            Yhat_1.append(mod.predict(np.c_[X[:,:-1], np.ones_like(A)], offset=offsets) if mod else np.full(Y.shape[0], np.mean(Y[:,j])))
        else:
            Yhat.append(mod.predict(X, offset=offsets) if mod else np.full(Y.shape[0], np.mean(Y[:,j])))

    families = {
        'gaussian': lambda disp: sm.families.Gaussian(),
        'poisson': lambda disp: sm.families.Poisson(),
        'nb': lambda disp: sm.families.NegativeBinomial(alpha=disp)
    }

    B = []
    d = X.shape[1]
    for j in range(Y.shape[1]):
        glm_family = families.get(family, lambda: ValueError('family must be one of: "gaussian", "poisson", "nb"'))(disp_glm[j] if family == 'nb' else None)
        try:
            mod = sm.GLM(Y[:,j], X, family=glm_family, offset=offsets).fit()
            B.append(mod.params)
            append_values(impute, Y, j, X, A, offsets, mod)
            tvals.append(mod.tvalues[-1])
            pvals.append(mod.pvalues[-1])
            resid_response.append(mod.resid_response)
        except:
            B.append(np.full(X.shape[1], np.nan))
            append_values(impute, Y, j, X, A, offsets)
            tvals.append(np.nan)
            pvals.append(np.nan)
            resid_response.append(np.full(Y.shape[0], np.mean(Y[:,j])))
    B = np.array(B)
    
    tvals = np.array(tvals)
    pvals = np.array(pvals)
    resid_response = np.array(resid_response).T

    if impute:
        Yhat_0 = np.array(Yhat_0).T
        Yhat_1 = np.array(Yhat_1).T
        Yhat = (Yhat_0, Yhat_1)
    else:
        Yhat = np.array(Yhat).T
    
    if return_df:
        df_glm = pd.DataFrame({
            'beta_hat':B[:,-1],
            'z_scores':tvals, 
            'p_values':pvals,
            'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
        })

        return df_glm
    else:
        return B, Yhat, tvals, pvals, resid_response


def glm_test(Y, X, A):
    '''
    Fit GLM to each column of Y, with covariate X and treatment A.

    Parameters
    ----------
    Y : array
        n x p matrix of outcomes
    X : array
        n x d matrix of covariates
    A : array
        n x 1 vector of treatments

    Returns
    -------
    df_glm_p, df_glm_nb : DataFrame
        results of GLM fit for each outcome, with p-values and q-values
    '''

    B_glm, _, tvals, pvals,_ = fit_glm(Y, X, A, 'poisson')
    df_glm_p = pd.DataFrame({
        'beta_hat':B_glm[:,-1],
        'z_scores':tvals, 
        'p_values':pvals,
        'q_values':multipletests(pvals, alpha=0.05, method='fdr_bh')[1]
    })


    mu_glm = np.mean(np.exp(np.c_[X,A] @ B_glm.T), axis=0)
    disp_glm = (np.mean((Y - mu_glm[None,:])**2, axis=0) - mu_glm) / mu_glm**2
    disp_glm = np.clip(disp_glm, 0.01, 100.)

    df_glm_nb = fit_glm(Y, X, A, 'nb', disp_glm, return_df=True)

    return df_glm_p, df_glm_nb