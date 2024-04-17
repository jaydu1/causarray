import os
import random
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import pprint
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)


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



def _geo_mean(x):
    non_zero_x = x[x != 0]
    if len(non_zero_x) == 0:
        return 0
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
        The desired library size after normalziation for 'scale'.
    
    Returns
    -------
    size_factor : array-like
        The size factors of the rows.
    '''
    if method=='geomeans':
        log_geo_means = np.apply_along_axis(_geo_mean, axis=0, arr=counts)
        log_size_factor = np.apply_along_axis(_normalize, axis=1, arr=Y, log_geo_means=log_geo_means)
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


