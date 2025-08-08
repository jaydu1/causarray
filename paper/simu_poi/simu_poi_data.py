import os
import sys
import numpy as np
import random
import scipy as sp
import scipy.linalg
from scipy.sparse.linalg import eigsh
from scipy.linalg import sqrtm
from tqdm import tqdm
from sklearn.datasets import make_classification
import h5py


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    np.random.seed(seed)
    random.seed(seed)


def generate_data(
    n_samples, n_features, d, r, s, 
    m=50, ct_num=3, ct_mean=.5, ct_scale=.5,
    signal=.2, noise=1., intercept=1.,
    psi=0.2, seed=0, scale_alpha=1., scale_beta=.5
    ):
    '''
    Parameters
    ----------
    n_samples: int
        Number of subject.
    n_features: int
        Number of features.
    d: int
        Number of latent variables.
    s: int
        Number of non-zero effects.
    m: int
        Number of cells per subject.
    ct_num: int
        Number of cell types.
    ct_mean: float
        Maximum difference in mean for different cell types.
    ct_scale: float
        Maximum difference in scale for different cell types.
    
    signal: float
        Signal strength.
    noise: float
        Noise level.
    intercept: float
        Intercept
    psi: float
        Probability of zero-inflation
    seed: int
        Random seed

    Returns
    -------
    W: np.ndarray
        Covariates (n_samples, d+1)
    A: np.ndarray
        Treatment assignment (n_samples, )
    Y: np.ndarray
        Pseudo-bulk expression data (n_samples x n_features)
    Y_sc: np.ndarray
        Single-cell expression data (n_samples x n_features)
    metadata: np.ndarray
        Metadata (n_samples x 4) for cell, indv, trt, and celltype.
    B: np.ndarray
        Coefficients (d+1, n_features)
    theta: np.ndarray
        True effects (n_features, )
    signal: np.ndarray
        True signals (n_features, )

    '''
    reset_random_seeds(seed)

    W = np.ones((n_samples,1))
    B = intercept * np.random.beta(2, 1, size=(1, n_features))
    
    ct = np.random.choice(ct_num, size=(n_samples,))
    ct_mean, ct_scale = np.linspace(-ct_mean, ct_mean, num=ct_num).reshape(-1,1), np.random.uniform(1-ct_scale, 1, size=(ct_num,1))
    ct_mean = ct_mean[ct]
    ct_scale = ct_scale[ct]

    if d > 0:
        # Generate covariates and treatment assignment
        _W = np.random.normal(size=(n_samples,d)) * 0.5
        _W[:,-r:] *= noise
        _W = (_W + ct_mean) * ct_scale
        W = np.c_[W, _W]

        beta = np.random.normal(size=(d+1,))/(2*np.sqrt(d))
        A = np.random.binomial(1, 1 / (1 + np.exp(- W @ beta)), size=(n_samples,))

        # Generate coefficients
        _B = np.random.normal(size=(d, n_features))/(2*np.sqrt(d))
        B = np.r_[B, _B]
    else:
        # W = (W + ct_mean) * ct_scale
        A = np.random.binomial(1, 0.5, size=(n_samples,))
    
    
    # Generate effects
    Theta = W @ B
    
    theta = np.zeros(n_features)
    theta[np.random.choice(np.arange(n_features), size=s, replace=False)] = 1.

    tmp = np.random.beta(scale_alpha, scale_beta, size=(1,n_features))
    signal = tmp * signal
    signal = signal * (np.random.binomial(1, 0.5, size=(1,n_features)) * 2 - 1)
    
    # Generate observations
    Theta = np.where(
        (A[:,None] @ theta[None,:])==1, Theta + signal, Theta
                )
    
    X = np.random.poisson(np.tile(np.exp(Theta), (m,1,1)))
    zero_inflation = np.random.binomial(1, 1-psi, X.shape)
    Y_sc = X * zero_inflation
    Y = np.sum(Y_sc, axis=0)

    
    metadata = np.c_[
        np.arange(Y_sc.shape[0]*Y_sc.shape[1]),
        np.tile(np.arange(Y_sc.shape[1]), Y_sc.shape[0]),
        np.tile(A, Y_sc.shape[0]),
        np.tile(ct, Y_sc.shape[0]),        
    ]
    Y_sc = Y_sc.reshape(-1, n_features)
    return W, A, Y, Y_sc, metadata, B, theta, signal





if __name__ == "__main__":
    # get params from command line in the format of '_d_X_r_Y_noise_Z'
    if len(sys.argv)>1:
        ind = str(sys.argv[1])
        r = int(ind.split('_')[4])
        d = int(ind.split('_')[2]) + r
        noise = float(ind.split('_')[-1])
    else:
        ind = ''
        ind = 0
        d = 0
        r = 0
        noise = 1.
    # n = 500 # number of cells
    # d = 0 # number of covariates
    p = 2000 # number of genes
    s = 200 # number of signals    
    signal = .5 # signal strength
    # noise = .5 # noise level
    psi = 0.1 # zero-inflation probability

    # Save data to HDF5 file
    for n in [100, 500, 1000, 5000]:
        path_to_data = '/home/jinandmaya/simu_poi/data/simu_{}{}/'.format(n, ind)
        os.makedirs(path_to_data, exist_ok=True)
        for seed in tqdm(range(50)):
            W, A, Y, Y_sc, metadata, B, theta, signals = generate_data(
                n, p, d, r, s, m=10, intercept=1., signal=signal, noise=noise, psi=psi, seed=seed)
            with h5py.File(path_to_data+'simu_data_{}.h5'.format(seed), 'w') as f:
                f.create_dataset('W', data=W)
                f.create_dataset('A', data=A)
                f.create_dataset('Y', data=Y)
                f.create_dataset('Y_sc', data=Y_sc)
                f.create_dataset('metadata', data=metadata)
                f.create_dataset('B', data=B)
                f.create_dataset('theta', data=theta)
                f.create_dataset('signal', data=signals)

