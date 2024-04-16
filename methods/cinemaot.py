import sys
sys.path.append("CINEMA-OT/")

import numpy as np
import pandas as pd
import cinemaot as co
import scanpy as sc
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def run_cinemaot(Y, A, raw=False, weighted=False, thres=0.15, smoothness=1e-3, **kwargs):
    '''
    The Python wrapper for running CINEMA-OT.

    Parameters
    ----------
    Y : np.ndarray
        The (preprocessed) gene expression matrix of shape [n,p].
    A : np.ndarray
        The treatment vector of shape [n,].
    raw : bool
        Whether the input data is raw or preprocessed.
    weighted : bool
        Whether to use the weighted version of CINEMA-OT.
    smoothness : float
        The smoothness parameter for CINEMA-OT.

    Returns
    -------
    de : np.ndarray
        The treatment effect matrix of shape [n,p].
    '''
    adata = sc.AnnData(Y.copy())
    A = np.array(A)
    if A.ndim > 1:
        A = A[:, 0]
    adata.obs['A'] = A

    # assuming the data is properly normalized
    if raw:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    sc.pp.pca(adata)
    if weighted:
        cf, ot, de, _ = co.cinemaot.cinemaot_weighted(
            adata, obs_label='A', ref_label=0, expr_label=1,
            smoothness=smoothness, **kwargs)
    else:
        cf, ot, de = co.cinemaot.cinemaot_unweighted(
            adata, obs_label='A', ref_label=0, expr_label=1,
            smoothness=smoothness, **kwargs)
    de = de.X
    
    # Y_hat_0 = Y.copy()
    # Y_hat_1 = Y.copy()    
    # for a in range(2):
    #     if weighted:
    #         _, ot, _de = co.cinemaot.cinemaot_weighted(
    #         adata, obs_label='A', ref_label=a, expr_label=1-a,
    #         smoothness=smoothness, **kwargs)
    #     else:
    #         _, ot, _de = co.cinemaot.cinemaot_unweighted(
    #             adata, obs_label='A', ref_label=a, expr_label=1-a,
    #             smoothness=smoothness, **kwargs)
    #     if a==0:
    #         Y_hat_0[A==a] = np.matmul(ot/np.sum(ot,axis=1)[:,None],Y[adata.obs['A']==1-a,:])
    #         de = _de
    #     else:
    #         Y_hat_1[A==a] = np.matmul(ot/np.sum(ot,axis=1)[:,None],Y[adata.obs['A']==1-a,:])
        # de[A==a] = (2 * a - 1) * _de.X

    stat, pvalue = list(zip(*[wilcoxon(de[:,j],zero_method='zsplit') for j in range(de.shape[1])]))
    # stat, pvalue = list(zip(*[wilcoxon((Y_hat_1-Y_hat_0)[:,j],zero_method='zsplit') for j in range(de.shape[1])]))
    padj = multipletests(pvalue, alpha=0.05, method='fdr_bh')[1]

    df = pd.DataFrame({'stat':stat, 'pvalue':pvalue, 'padj':padj})
    return {'cinemaot.df':df, 'cinemaot.res':{'W':cf, 'cf':np.log1p(Y[A==0]), 'de':de}}
