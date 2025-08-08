import sys
sys.path.append("CINEMA-OT/")

import numpy as np
import pandas as pd
import cinemaot as co
import scanpy as sc
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn.neighbors import NearestNeighbors


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
    
    Y_hat_0 = np.zeros(Y.shape, dtype=float)
    Y_hat_1 = np.zeros(Y.shape, dtype=float)
    de = np.zeros(Y.shape, dtype=float)
    for a in range(2):
        if weighted:
            _cf, ot, _de, _ = co.cinemaot.cinemaot_weighted(
            adata, obs_label='A', ref_label=a, expr_label=1-a,
            smoothness=smoothness, **kwargs)
        else:
            _cf, ot, _de = co.cinemaot.cinemaot_unweighted(
                adata, obs_label='A', ref_label=a, expr_label=1-a,
                smoothness=smoothness, **kwargs)

        if a==0:
            Y_hat_0[A==a] -= _de.X
            cf = _cf
        else:
            Y_hat_1[A==a] -= _de.X
        de[A==a] = (2 * a - 1) * _de.X

    stat, pvalue = list(zip(*[wilcoxon(de[:,j],zero_method='zsplit') for j in range(de.shape[1])]))
    padj = multipletests(pvalue, alpha=0.05, method='fdr_bh')[1]

    df = pd.DataFrame({'stat':stat, 'pvalue':pvalue, 'padj':padj})
    return {'cinemaot.df':df, 'cinemaot.res':{'W':cf, 'Y_hat_0':Y_hat_0, 'Y_hat_1':Y_hat_1, 'de':de}}



def run_mixscape(Y, A, raw=False, nn=20, **kwargs):
    '''
    The Python wrapper for running Mixscape.

    Parameters
    ----------
    Y : np.ndarray
        The (preprocessed) gene expression matrix of shape [n,p].
    A : np.ndarray
        The treatment vector of shape [n,].
    raw : bool
        Whether the input data is raw or preprocessed.
    nn : int
        The number of nearest neighbors to use.

    Returns
    -------
    A dictionary containing the results dataframe and other matrices.
    '''
    adata = sc.AnnData(Y.copy())
    if A.ndim > 1:
        A = A[:, 0]
    adata.obs['A'] = A

    if raw:
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    
    sc.pp.pca(adata)

    Y_hat_0 = np.zeros_like(Y)
    Y_hat_1 = np.zeros_like(Y)
    de = np.zeros_like(Y)

    # Calculate counterfactual for treated group (A=1)
    X_pca_ctrl = adata.obsm['X_pca'][adata.obs['A'] == 0, :]
    X_pca_trt = adata.obsm['X_pca'][adata.obs['A'] == 1, :]
    nbrs_ctrl = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X_pca_ctrl)
    mixscape_matrix_trt = nbrs_ctrl.kneighbors_graph(X_pca_trt).toarray()
    
    Y_hat_0[A == 1] = (mixscape_matrix_trt / np.sum(mixscape_matrix_trt, axis=1, keepdims=True)) @ Y[A == 0]
    Y_hat_0[A == 0] = Y[A == 0]

    # Calculate counterfactual for control group (A=0)
    X_pca_trt = adata.obsm['X_pca'][adata.obs['A'] == 1, :]
    X_pca_ctrl = adata.obsm['X_pca'][adata.obs['A'] == 0, :]
    nbrs_trt = NearestNeighbors(n_neighbors=nn, algorithm='ball_tree').fit(X_pca_trt)
    mixscape_matrix_ctrl = nbrs_trt.kneighbors_graph(X_pca_ctrl).toarray()

    Y_hat_1[A == 0] = (mixscape_matrix_ctrl / np.sum(mixscape_matrix_ctrl, axis=1, keepdims=True)) @ Y[A == 1]
    Y_hat_1[A == 1] = Y[A == 1]

    de[A == 1] = Y[A == 1] - Y_hat_0[A == 1]
    de[A == 0] = Y_hat_1[A == 0] - Y[A == 0]

    stat, pvalue = list(zip(*[wilcoxon(de[:, j], zero_method='zsplit') for j in range(de.shape[1])]))
    padj = multipletests(pvalue, alpha=0.05, method='fdr_bh')[1]

    df = pd.DataFrame({'stat': stat, 'pvalue': pvalue, 'padj': padj})
    
    # The concept of a coupling matrix 'W' is specific to OT, returning None.
    return {'mixscape.df': df, 'mixscape.res': {'W': adata.obsm['X_pca'], 'Y_hat_0': Y_hat_0, 'Y_hat_1': Y_hat_1, 'de': de}}