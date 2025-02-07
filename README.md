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

For `R` users, `reticulate` can be used to call `causarray` from `R`.
The documentation and tutorials using both `Python` and `R` are available at [causarray.readthedocs.io](https://causarray.readthedocs.io/en/latest/).



## Logs

- [x] (2025-01-30) Python package released on PyPI
- [x] (2025-02-01) code for reproducing figures in paper
- [x] (2025-02-02) Tutorial for Python and R
- [ ] Documentation


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
