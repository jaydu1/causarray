[![Documentation Status](https://readthedocs.org/projects/causarray/badge/?version=latest)](https://causarray.readthedocs.io/en/latest/?badge=latest)
[![PyPI](https://img.shields.io/pypi/v/causarray?label=pypi)](https://pypi.org/project/causarray)
[![PyPI-Downloads](https://img.shields.io/pepy/dt/causarray)](https://pepy.tech/project/causarray)


# causarray

Advances in single-cell sequencing and CRISPR technologies have enabled detailed case-control comparisons and experimental perturbations at single-cell resolution. However, uncovering causal relationships in observational genomic data remains challenging due to selection bias and inadequate adjustment for unmeasured confounders, particularly in heterogeneous datasets. To address these challenges, we introduce `causarray` [Du25], a doubly robust causal inference framework for analyzing array-based genomic data at both bulk-cell and single-cell levels. `causarray` integrates a generalized confounder adjustment method to account for unmeasured confounders and employs semiparametric inference with ﬂexible machine learning techniques to ensure robust statistical estimation of treatment effects.


## Usage

We recommend using `causarray` in a conda environment:
```cmd
# create a new conda environment and install the necessary packages
conda create -n causarray python=3.12 -y

# activate the environment
conda activate causarray
```

The module can be installed via PyPI:
```cmd
pip install causarray
```

For optimal parallel performance, we recommend installing `llvm-openmp` if using conda:
```cmd
conda install -c conda-forge llvm-openmp
```

For `R` users, `reticulate` can be used to call `causarray` from `R`.
The documentation and tutorials using both `Python` and `R` are available at [causarray.readthedocs.io](https://causarray.readthedocs.io/en/latest/).



## Batch fitting for large-scale screens

For screens with hundreds to thousands of perturbations, use the batch API
so that peak memory is bounded by one batch at a time:

```python
from causarray import gcate_lfc_batch

df_res = gcate_lfc_batch(
    Y, X, A, r,
    batch_size=10,    # perturbations per batch (or use n_batches= for a fixed count)
    max_cells=2000,   # max pert cells per batch (ctrl added on top)
    n_ctrl=2000,      # fixed ctrl subsample shared across batches
    cache_path='results.h5',   # resume if interrupted
    verbose=True,
)
```

See the [Replogle-E-K562 tutorial](https://causarray.readthedocs.io/en/latest/)
for a demonstration on 200 perturbations from a genome-wide CRISPRi screen.

## Changelog

- [x] (2025-01-30) Python package released on PyPI
- [x] (2025-02-01) Code for reproducing figures in paper
- [x] (2025-02-02) Tutorial for Python and R
- [x] (2026-05-31) Batch fitting API (`gcate_lfc_batch`) for large-scale screens
- [x] (2026-05-31) Documentation at [causarray.readthedocs.io](https://causarray.readthedocs.io/en/latest/)


<!-- 
# Development

The dependencies for running `causarray` method are listed in `environment.yml` and can be installed by running

```cmd
PIP_NO_DEPS=1 conda env create -f environment.yml
```


## Build
```cmd
git tag 0.0.0
git tag --delete 1.0.0
python -m pip install .
```

## Testing
```cmd
python -m pytest tests/test_gcate.py
python -m pytest tests/test_DR_learner.py
```

## Documentation

```cmd
mkdir docs
cd docs
sphinx-quickstart

make html # sphinx-build source build


rmarkdown::render("perturbseq.Rmd", rmarkdown::md_document(variant = "markdown_github"))
```
-->


## References
[Du25] Jin-Hong Du, Maya Shen, Hansruedi Mathys, and Kathryn Roeder (2025). Causal differential expression analysis under unmeasured confounders with causarray. bioRxiv, 2025-01.
