# causarray

Advances in single-cell sequencing and CRISPR technologies have enabled detailed case-control comparisons and experimental perturbations at single-cell resolution. However, uncovering causal relationships in observational genomic data remains challenging due to selection bias and inadequate adjustment for unmeasured confounders, particularly in heterogeneous datasets. To address these challenges, we introduce `causarray` [Du25], a doubly robust causal inference framework for analyzing array-based genomic data at both bulk-cell and single-cell levels. `causarray` integrates a generalized confounder adjustment method to account for unmeasured confounders and employs semiparametric inference with ﬂexible machine learning techniques to ensure robust statistical estimation of treatment effects.


# Requirements

The dependencies for running `causarray` method are listed in `environment.yml` and can be installed by running

```cmd
PIP_NO_DEPS=1 conda env create -f environment.yml
```




<!-- 
# Development

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
sphinx-quickstart
cd docs
make html # sphinx-build source build
```
-->


# References
[Du25] Jin-Hong Du, Maya Shen, Hansruedi Mathys, and Kathryn Roeder (2025). Causal differential expression analysis under unmeasured confounders with causarray. bioRxiv, 2025-01.