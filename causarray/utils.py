import os
import random
import numpy as np
import pandas as pd

import inspect

import pprint
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

np.set_printoptions(threshold=10)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot


def prep_causarray_data(Y, A, X=None, X_A=None, intercept=True):
    """
    Prepares the input data for the causarray model.

    Parameters
    ----------
    Y : array-like
        The response matrix.
    A : array-like
        The treatment matrix.
    X : array-like, optional
        The covariate matrix. Defaults to None.
    X_A : array-like, optional
        The covariate matrix for the treatment. Defaults to None.
    intercept : bool, optional
        Whether to include an intercept in the covariate matrix. Defaults to True.

    Returns
    -------
    Y : array
        The processed response matrix.
    A : array
        The processed treatment matrix.
    X : array
        The processed covariate matrix.
    X_A : array
        The processed covariate matrix with the log library size.
    """
    if not isinstance(Y, pd.DataFrame):
        Y = np.asarray(Y)
    Y = np.minimum(Y, np.round(np.quantile(np.max(Y, 0), 0.999)))
    if not isinstance(A, pd.DataFrame):
        A = np.asarray(A)

    X = np.zeros((Y.shape[0], 0)) if X is None else np.asarray(X)    
    intercept_col = np.ones((X.shape[0], 1)) if intercept else np.empty((X.shape[0], 0))
    X = np.hstack((intercept_col, X))

    X_A = X if X_A is None else np.asarray(X_A)
    loglibsize = np.log2(np.sum(np.asarray(Y), axis=1))
    loglibsize = (loglibsize - np.mean(loglibsize)) / np.std(loglibsize, ddof=1)
    X_A = np.hstack((X_A, loglibsize[:, None]))

    return Y, A, X, X_A


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)


def _filter_params(func, kwargs):
    '''
    Filter the parameters of a function.

    Parameters
    ----------
    func : function
        The function to filter the parameters.
    kwargs : dict
        The input parameters.

    Returns
    -------
    filtered_kwargs : dict
        The filtered parameters.
    '''
    if isinstance(func, dict):
        valid_params = func.keys()
    elif callable(func):
        valid_params = inspect.signature(func).parameters.keys()
    else:
        raise ValueError("The provided func is not a callable function, or a dict.")
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    
    return filtered_kwargs


class Early_Stopping():
    '''
    The early-stopping monitor.
    '''
    def __init__(self, warmup=25, patience=25, tolerance=0., is_minimize=True, **kwargs):
        self.warmup = warmup
        self.patience = patience
        self.tolerance = tolerance
        self.is_minimize = is_minimize

        self.step = -1
        self.best_step = -1
        self.best_metric = np.inf

        if not self.is_minimize:
            self.factor = -1.0
        else:
            self.factor = 1.0
        self.info = None

    def __call__(self, metric):
        self.step += 1
        
        if self.step < self.warmup:
            return False
        elif self.factor*metric<self.factor*self.best_metric-self.tolerance:
            self.best_metric = metric
            self.best_step = self.step
            return False
        elif self.step - self.best_step>self.patience:
            self.info = 'Best Epoch: %d. Best Metric: %f.'%(self.best_step, self.best_metric)
            return True
        else:
            return False

    def reset_state(self):
        self.best_step = self.step
        self.best_metric = np.inf



def _geo_mean(x):
    non_zero_x = x[x != 0]
    if len(non_zero_x) == 0:
        return -np.inf
    else:
        return np.mean(np.log(non_zero_x))
        
def _normalize(counts, log_geo_means):
    log_cnts = np.log(counts)
    diff = log_cnts - log_geo_means
    mask = np.isfinite(log_geo_means) & (counts > 0)
    return np.median(diff[mask])


def comp_size_factor(counts, method='geomeans', lib_size=1e4, **kwargs):
    '''
    Compute the size factors of the rows of the count matrix.

    Parameters
    ----------
    counts : array-like
        The input raw count matrix.
    method : str
        The method to compute the size factors, 'geomeans' or 'scale'.
    lib_size : float
        The desired library size after normalization for 'scale'.
    
    Returns
    -------
    size_factor : array-like
        The size factors of the rows.
    '''
    if method=='geomeans':
        # compute the geometric mean of all genes
        log_geo_means = np.apply_along_axis(_geo_mean, axis=0, arr=counts)
        log_size_factor = np.apply_along_axis(_normalize, axis=1, arr=counts, log_geo_means=log_geo_means)
        size_factor = np.exp(log_size_factor - np.mean(log_size_factor))
    elif method=='scale':
        size_factor = 1./np.sum(Y, axis=0)*lib_size
    else:
        raise ValueError("Method must be in {'geomeans' or 'scale'}.")

    return size_factor



def plot_r(df_r, c=1):
    '''
    Plot the results of the estimation of the number of latent factors.

    Parameters
    ----------
    df_r : DataFrame
        Results of the number of latent factors.
    c : float
        The constant factor for the complexity term.

    Returns
    -------
    fig : Figure
        The figure of the plot.
    '''
    
    
    fig = plt.figure(figsize=[18,6])
    host = host_subplot(121)
    par = host.twinx()

    host.set_xlabel("Number of factors $r$")
    host.set_ylabel("Deviance")
    # par.set_ylabel("$\nu$")


    p1, = host.plot(df_r['r'], df_r['deviance'], '-o', label="Deviance")
    p2, = par.plot(df_r['r'], df_r['nu']*c, '-o', label=r"$\nu$")


    host.set_xticks(df_r['r'])
    host.yaxis.get_label().set_color(p1.get_color())
    par.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=14)

    p1, = host.plot(df_r['r'], df_r['deviance']+df_r['nu']*c, '-o', label="JIC")
    host.legend(labelcolor="linecolor")


    host = host_subplot(122)
    par = host.twinx()
    host.set_xlabel("Number of factors $r$")
    par.set_ylabel(r"$\nu$")

    p1, = host.plot(df_r['r'].iloc[1:], -np.diff(df_r['deviance']), '-o', label='diff dev')
    p2, = par.plot(df_r['r'].iloc[1:], np.diff(df_r['nu'])*c,  '-o', label=r'diff $\nu$')

    host.legend(labelcolor="linecolor")
    host.set_xticks(df_r['r'].iloc[1:])
    par.set_ylim(*host.get_ylim())
    
    par.yaxis.get_label().set_color(p2.get_color())
    par.tick_params(axis='y', colors=p2.get_color(), labelsize=14)
    host.tick_params(axis='y', colors=p1.get_color(), labelsize=14)

    return fig


