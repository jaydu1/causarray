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