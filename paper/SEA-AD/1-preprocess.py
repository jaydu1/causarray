# %%
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# %% [markdown]
# # MTG

# %%
adata = sc.read_h5ad('MTG.h5ad')
adata = adata[adata.obs['Subclass'].isin(['L2/3 IT', 'L6 IT', 'L4 IT', 'L5 IT', 'L5 ET', 'L6 CT', 'L6b', 'L6 IT Car3', 'L5/6 NP'])]
adata.write_h5ad('MTG_exneu_sc.h5ad', compression='gzip')
adata

# %%
# Initialize an empty list to store subsampled indices
subsampled_indices = []

# Loop through each unique donor_id and sample 300 cells
for donor in adata.obs['donor_id'].unique():
    donor_indices = adata.obs[adata.obs['donor_id'] == donor].index
    if len(donor_indices) > 300:
        sampled_indices = np.random.choice(donor_indices, 300, replace=False)
    else:
        sampled_indices = donor_indices
    subsampled_indices.extend(sampled_indices)

# Convert the list to a numpy array
subsampled_indices = np.array(subsampled_indices)


# %%
obs['Subclass'].value_counts().index

# %%

fig, axes = plt.subplots(1,2,figsize=(10, 4))

sns.countplot(data=adata.obs, x='Subclass', order=adata.obs['Subclass'].value_counts().index, ax=axes[0])

sns.countplot(data=adata.obs.loc[subsampled_indices], x='Subclass', order=adata.obs['Subclass'].value_counts().index, ax=axes[1])

for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Subclass')
axes[0].set_ylabel('Count')
axes[1].set_ylabel('')
axes[0].set_title('Full dataset')
axes[1].set_title('Subsample dataset')

# %%
subsampled_indices

# %%
adata = adata[subsampled_indices]
adata.write_h5ad('MTG_exneu_sc_subsample.h5ad', compression='gzip')

# %%
adata

# %%
'Age at death', 'Cognitive status', 'ADNC',
'Braak stage', 'Thal phase', 'CERAD score', 'APOE4 status',
'Lewy body disease pathology', 'LATE-NC stage',
'Microinfarct pathology', 'Specimen ID', 'donor_id', 'PMI',       
'suspension_type', 'development_stage_ontology_term_id',
'Continuous Pseudo-progression Score', 'tissue_type', 'cell_type',
'assay', 'disease', 'sex'

# %%
obs['Cognitive status'].unique()

# %%
len(obs['donor_id'].unique())

# %% [markdown]
# # PFC

# %%
adata = sc.read_h5ad('PFC.h5ad')
adata = adata[adata.obs['Subclass'].isin(['L2/3 IT', 'L6 IT', 'L4 IT', 'L5 IT', 'L5 ET', 'L6 CT', 'L6b', 'L6 IT Car3', 'L5/6 NP'])]
adata.write_h5ad('PFC_exneu_sc.h5ad', compression='gzip')
adata

# %%
# Initialize an empty list to store subsampled indices
subsampled_indices = []

# Loop through each unique donor_id and sample 300 cells
for donor in adata.obs['donor_id'].unique():
    donor_indices = adata.obs[adata.obs['donor_id'] == donor].index
    if len(donor_indices) > 300:
        sampled_indices = np.random.choice(donor_indices, 300, replace=False)
    else:
        sampled_indices = donor_indices
    subsampled_indices.extend(sampled_indices)

# Convert the list to a numpy array
subsampled_indices = np.array(subsampled_indices)


# %%

fig, axes = plt.subplots(1,2,figsize=(10, 4))

sns.countplot(data=adata.obs, x='Subclass', order=adata.obs['Subclass'].value_counts().index, ax=axes[0])

sns.countplot(data=adata.obs.loc[subsampled_indices], x='Subclass', order=adata.obs['Subclass'].value_counts().index, ax=axes[1])

for ax in axes:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set_xlabel('Subclass')
axes[0].set_ylabel('Count')
axes[1].set_ylabel('')
axes[0].set_title('Full dataset')
axes[1].set_title('Subsample dataset')

# %%
adata = adata[subsampled_indices]
adata.write_h5ad('PFC_exneu_sc_subsample.h5ad', compression='gzip')

# %%
adata

# %%
'Age at death', 'Cognitive status', 'ADNC',
'Braak stage', 'Thal phase', 'CERAD score', 'APOE4 status',
'Lewy body disease pathology', 'LATE-NC stage',
'Microinfarct pathology', 'Specimen ID', 'donor_id', 'PMI',       
'suspension_type', 'development_stage_ontology_term_id',
'Continuous Pseudo-progression Score', 'tissue_type', 'cell_type',
'assay', 'disease', 'sex'

# %%
adata.obs['Cognitive status'].unique()

# %%
len(adata.obs['donor_id'].unique())


