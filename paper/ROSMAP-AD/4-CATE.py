import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import scipy as sp
import h5py
import pickle

# Add the methods directory to the system path
sys.path.append('../methods/')
from causarray import LFC
from causarray.DR_estimation import AIPW_mean
from causarray.DR_learner import VIM
from statsmodels.stats.multitest import multipletests

# Define the cell type and propensity score model
celltype = 'exneu'
ps_model = 'random_forest_cv'

# Define the path to the results directory
path = 'results/{}/DE/data/'.format(ps_model)

# Load the data from CSV files
Y = pd.read_csv(path+"Y.csv", index_col=0).values
W = pd.read_csv(path+"Wp.csv", index_col=0).values
W_raw = pd.read_csv(path+"W.raw.csv", index_col=0).values
A = pd.read_csv(path+"A.csv", index_col=0).values
gene_names = pd.read_csv(path+"Y.csv", index_col=0).columns

# Normalize the log library sizes
logls = np.log2(np.sum(Y, axis=1))
logls = (logls - np.mean(logls))/np.std(logls, ddof=1)
W_A = np.c_[W, logls]

# Perform the LFC analysis
df, res = LFC(Y, W, A, W_A=W_A, ps_model=ps_model, offset=True, verbose=True)
df['gene_names'] = gene_names

# Print the number of significant genes
print(np.sum(df['padj']<0.1))

# Extract results from the LFC analysis
pi = res['pi']
Y_hat_0 = res['Y_hat_0']
Y_hat_1 = res['Y_hat_1']
size_factors = res['size_factors']

# Get the number of samples and genes
n, p = Y.shape


_, eta_0 = AIPW_mean(Y.astype('float'), 1-A, np.clip(Y_hat_0, None, 1e5), 1-pi, pseudo_outcome=True, positive=True)
_, eta_1 = AIPW_mean(Y.astype('float'), A, np.clip(Y_hat_1, None, 1e5), pi, pseudo_outcome=True, positive=True)
eta_0 = eta_0[:,:,0]/size_factors[:,None]
eta_1 = eta_1[:,:,0]/size_factors[:,None]
thres_diff = 1e-6

tau_0, tau_1 = np.mean(eta_0, axis=0), np.mean(eta_1, axis=0)
tau_1 = np.clip(tau_1, thres_diff, None)
tau_0 = np.clip(tau_0, thres_diff, None)

tau_estimate = np.log(tau_1/tau_0)
eta = eta_1 / tau_1[None,:] -  eta_0 / tau_0[None,:]
theta_var = np.var(eta, axis=0, ddof=1)

# filter out low-expressed genes
idx = ~ ((np.maximum(tau_0,tau_1)<1e-4) & ((tau_1-tau_0)<1e-6))
tau_estimate[~idx] = 0.; eta[:,~idx] = 0.; theta_var[~idx] = np.inf


sqrt_theta_var = np.sqrt((theta_var + 1e-3)/ n)
tvalues_init = tau_estimate / sqrt_theta_var
phi = eta + tau_estimate[None,:]


# # Point estimation of the treatment effect
# tau_0, eta_0 = AIPW_mean(np.log1p(Y / size_factors[:,None]), 1-A, np.log1p(Y_hat_0 / size_factors[:,None,None]), 1-pi, pseudo_outcome=True, positive=True)
# tau_1, eta_1 = AIPW_mean(np.log1p(Y / size_factors[:,None]), A, np.log1p(Y_hat_1 / size_factors[:,None,None]), pi, pseudo_outcome=True, positive=True)

# # Calculate the treatment effect estimates
# tau_estimate = (tau_1 - tau_0)[:,0]
# eta = (eta_1 - eta_0)[:,:,0] - tau_estimate[None, :]
# phi = (eta_1 - eta_0)[:,:,0]

# Calculate t-values, p-values, and q-values
# tvalues_init = np.sqrt(n) * (tau_estimate) / np.sqrt(np.var(eta, axis=0, ddof=1))
# pvals = sp.stats.norm.sf(np.abs(tvalues_init)) * 2
# qvals = multipletests(pvals, alpha=0.05, method='fdr_bh')[1]

# Define the number of covariates to exclude
r = 10
d = W.shape[1] - r

# Identify cells with less than 90 counts
id_cells = W_raw[:,-1] < 90

# Perform variable importance measurement (VIM)
res_vim = VIM(phi[id_cells], W[id_cells], id_covs=np.arange(2,d), degree=1)

# Print summary statistics
print(len(df[df['padj']<0.1].index), res_vim['CATE'].shape, W_raw.shape, np.sum(qvals<0.1))

# Define the objects to be saved
objects_to_save = {
    'df': df,
    'W_raw': W_raw,
    'eta': eta,
    'phi': phi,
    'eta_1': eta_1,
    'eta_0': eta_0,
    # 'tvalues_init': tvalues_init,
    # 'pvals': pvals,
    # 'qvals': qvals,
    'res_vim': res_vim
}

# Save the objects to a pickle file
with open(path+'result_CATE.pkl', 'wb') as file:
    pickle.dump(objects_to_save, file)