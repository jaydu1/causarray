# Requirements

The dependencies for running `causarray` method are listed in `environment.yml` and can be installed by running

```cmd
PIP_NO_DEPS=1 conda env create -f environment.yml
```




<!-- 
# Development

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