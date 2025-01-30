from causarray.utils import *
from causarray.utils import _filter_params
import numpy as np
import statsmodels as stats
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from tqdm import tqdm

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



def fit_glm(Y, X, A=None, family='gaussian', disp_family='poisson',
    disp_glm=None, impute=False, offset=None, shrinkage=False,
    alpha=1e-4, maxiter=1000, thres_disp=100., n_jobs=-3, random_state=0, verbose=False, **kwargs):
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
    np.random.seed(random_state)
    
    if family not in ['gaussian', 'poisson', 'nb']:
        raise ValueError('Family not recognized')
   
    d = X.shape[1]

    if A is None:
        a = 1 # dummy treatment
        assert impute is False
    else:
        if A.ndim==1:
            A = A[:,None]
        if impute is not False and isinstance(impute, np.ndarray):
            X_test = impute
        else:
            X_test = X
        X_test = np.c_[X,np.zeros_like(A)]
        X = np.c_[X,A]
        # X_test = X.copy()

        a = A.shape[1]

    if offset is not None and offset is not False:
        if type(offset)==bool and offset is True:
            offsets = np.log(comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs)))
        else:
            offsets = offset
    else:
        offsets = None

    # estimate dispersion parameter for negative binomial GLM if not provided
    if family=='nb' and disp_glm is None:
        disp_glm = estimate_disp(Y, X, offset=offsets, disp_family=disp_family, maxiter=1000, verbose=verbose, **kwargs)
    
    alpha = np.full(X.shape[1], alpha)
    pprint.pprint('Fitting {} GLM{}...'.format(family, '' if offsets is None else ' with offset'))
    is_constant = np.all(X == X[0, :], axis=0)
    alpha[is_constant] = 0
    # alpha[:-a] = 0

    families = {
        'gaussian': lambda disp: sm.families.Gaussian(),
        'poisson': lambda disp: sm.families.Poisson(),
        'nb': lambda disp: sm.families.NegativeBinomial(alpha=1/disp)
    }

    def fit_model(j, Y, X, offsets, family, disp, impute, alpha):
        if family=='nb' and disp[j]>thres_disp:
            family = 'poisson'
        try:            
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                glm_family = families.get(family, lambda: ValueError('family must be one of: "gaussian", "poisson", "nb"'))(disp_glm[j] if family == 'nb' else None)

                try:
                    if shrinkage:
                        raise ValueError('fit regularized GLM')
                    mod = sm.GLM(Y[:,j], X, family=glm_family, offset=offsets).fit(maxiter=maxiter)
                    if not np.all(np.isfinite(mod.params)) or np.any(np.abs(mod.params[:d])>50) or np.any(np.abs(mod.params[d:])>10):
                        raise ValueError('GLM did not converge')
                    resid_deviance = mod.resid_deviance
                except:
                    mod = sm.GLM(Y[:,j], X, family=glm_family, offset=offsets).fit_regularized(alpha=alpha, cnvrg_tol=1e-5)
                    resid_deviance = np.full(Y.shape[0], 0.)

            B = mod.params

            Yhat_0 = np.zeros((Y.shape[0], a))
            Yhat_1 = np.zeros((Y.shape[0], a))
            if impute is not False:
                for j in range(a):
                    X_test_copy = X_test.copy()
                    Yhat_0[:,j] = mod.predict(X_test_copy, offset=offsets)                    
                    X_test_copy[:, d+j] = 1  # Update the j-th column with all ones
                    Yhat_1[:,j] = mod.predict(X_test_copy, offset=offsets)
            else:
                Yhat_0[:,:] = Yhat_1[:,:] = mod.predict(X, offset=offsets).reshape(-1, a)
            
        except:
            pprint.pprint('Fitting GLM for column {} does not converge.'.format(j))
            B = np.full(X.shape[1], 0.)
            
            Yhat_0 = np.full((Y.shape[0], a), 0)
            Yhat_1 = np.full((Y.shape[0], a), 0)
            if impute is not False:
                for k in range(a):
                    Yhat_0[:, k] = np.mean(Y[A[:, k] == 1, j])
                    Yhat_1[:, k] = np.mean(Y[A[:, k] == 0, j])
            resid_deviance = np.full(Y.shape[0], 0.)
        return B, Yhat_0, Yhat_1, resid_deviance


    results = Parallel(n_jobs=n_jobs)(delayed(fit_model)(
        j, Y, X, offsets, family, disp_glm, impute, alpha) for j in tqdm(range(Y.shape[1])))

    B, Yhat_0, Yhat_1, resid_deviance = zip(*results)
    B = np.array(B)    
    Yhat_0 = np.array(Yhat_0).transpose(1, 0, 2)
    Yhat_1 = np.array(Yhat_1).transpose(1, 0, 2)
    resid_deviance = np.array(resid_deviance).T

    if impute is not False:        
        Yhat = (Yhat_0, Yhat_1)
    else:
        Yhat = np.array(Yhat_0)[:,:,0]
    
    return B, Yhat, disp_glm, offsets, resid_deviance


def estimate_disp(Y, X=None, A=None, Y_hat=None, disp_family='gaussian', offset=None, verbose=False, **kwargs):
    if offset is not None:
        if type(offset)==bool and offset is True:
            offsets = np.log(comp_size_factor(Y, **_filter_params(comp_size_factor, kwargs)))
        else:
            offsets = offset
        sf = np.exp(offsets)[:,None]
    else:
        offsets = None
        sf = 1.
    

    if Y_hat is None:        
        if verbose:
            pprint.pprint('Estimating dispersion parameter...')

        if A is not None:
            X = np.c_[X,A]

        if disp_family=='gaussian':
            Y_norm = Y/sf
            reg = LinearRegression(fit_intercept=False).fit(X, Y_norm)
            Y_hat = reg.predict(X)
            # Y_hat = np.clip(Y_hat, 0, 1) * sf            
        elif disp_family=='poisson':
            Y_hat = fit_glm(Y, X, None, offset=offsets, family='poisson', impute=False, **kwargs)[1]      
            Y_hat /= sf

    # Clip Y_hat based on the range of Y per column
    Y_hat = np.clip(Y_hat, 0., np.max(Y/sf, axis=0))

    disp_glm = np.mean((Y/sf - Y_hat)**2 - Y_hat, axis=0) / np.mean(Y_hat**2, axis=0)
    disp_glm = 1./np.clip(disp_glm, 0.01, 100.)
    disp_glm[np.isnan(disp_glm)] = 1.

    return disp_glm




def loess_fit(Y, X, n_jobs=-3, **kwargs):
    
    def _loess_fit(y, x, **kwargs):
        try:
            from skmisc.loess import loess
            l = loess(x, y, **kwargs)
            l.fit()
            pred = l.predict(x, stderror=True)
            conf = pred.confidence()
            pred, lower, upper = pred.values, conf.lower, conf.upper
        except:
            pred, lower, upper = np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan)

        return pred, lower, upper

    results = Parallel(n_jobs=n_jobs)(delayed(_loess_fit)(Y[:,j], X, **kwargs) for j in range(Y.shape[1]))

    CATE, CATE_lower, CATE_upper = zip(*results)
    CATE = np.array(CATE).T
    CATE_lower = np.array(CATE_lower).T
    CATE_upper = np.array(CATE_upper).T
    return CATE, CATE_lower, CATE_upper



def ls_fit(Y, X, n_jobs=-3, **kwargs):
    
    def _ls_fit(y, x, **kwargs):
        # try:
        model = sm.OLS(y, x)
        result = model.fit(disp=False)

        # Get the predicted values
        pred = result.predict(x)

        # Get the confidence intervals
        conf = result.conf_int()    
        pred, lower, upper = pred, np.full(y.shape[0], conf[0][0]), np.full(y.shape[0], conf[0][1])
        # except:
        #     pred, lower, upper = np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan), np.full(y.shape[0], np.nan)

        return pred, lower, upper

    results = Parallel(n_jobs=n_jobs)(delayed(_ls_fit)(Y[:,j], X, **kwargs) for j in range(Y.shape[1]))

    CATE, CATE_lower, CATE_upper = zip(*results)
    CATE = np.array(CATE).T
    CATE_lower = np.array(CATE_lower).T
    CATE_upper = np.array(CATE_upper).T
    return CATE, CATE_lower, CATE_upper

    
    