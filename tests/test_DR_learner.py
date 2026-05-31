import numpy as np
import pandas as pd
import pytest
from causarray.DR_learner import LFC

# package/tests/test_DR_learner.py


@pytest.fixture
def sample_data():
    np.random.seed(0)
    Y = np.random.poisson(5, (100, 10))
    W = np.random.normal(0, 1, (100, 5))
    A = np.random.binomial(1, 0.5, 100)
    Y += np.random.poisson(1, (100, 10)) * A[:, None]
    return Y, W, A

def test_LFC_basic(sample_data):
    Y, W, A = sample_data
    result, estimation = LFC(Y, W, A)
    assert isinstance(result, pd.DataFrame)
    assert 'tau' in result.columns
    assert 'std' in result.columns
    assert 'stat' in result.columns
    assert 'rej' in result.columns
    assert 'pvalue' in result.columns
    assert 'padj' in result.columns
    assert 'pvalue_emp_null_adj' in result.columns
    assert 'padj_emp_null_adj' in result.columns

def test_LFC_with_offset(sample_data):
    Y, W, A = sample_data
    offset = np.log(np.random.poisson(5, 100))
    result, estimation = LFC(Y, W, A, offset=offset)
    assert isinstance(result, pd.DataFrame)
    assert 'tau' in result.columns

# def test_LFC_with_cross_est(sample_data):
#     Y, W, A = sample_data
#     result, estimation = LFC(Y, W, A, cross_est=True)
#     assert isinstance(result, pd.DataFrame)
#     assert 'tau' in result.columns

def test_LFC_with_fdx(sample_data):
    Y, W, A = sample_data
    result, estimation = LFC(Y, W, A, fdx=True)
    assert isinstance(result, pd.DataFrame)
    assert 'tau' in result.columns

def test_LFC_with_custom_family(sample_data):
    Y, W, A = sample_data
    result, estimation = LFC(Y, W, A, family='poisson')
    assert isinstance(result, pd.DataFrame)
    assert 'tau' in result.columns


def test_LFC_multi_treatment():
    """Test LFC with multiple perturbations (exercises per-perturbation fast path)."""
    np.random.seed(42)
    n, p, a = 200, 50, 3
    W = np.random.normal(0, 1, (n, 3))
    # One-hot treatment: first 50 cells = control, rest split among 3 perturbations
    A = np.zeros((n, a))
    A[50:100, 0] = 1
    A[100:150, 1] = 1
    A[150:200, 2] = 1
    Y = np.random.poisson(5, (n, p))
    # Add treatment effects for perturbation 0
    Y[50:100] += np.random.poisson(2, (50, p))

    result, estimation = LFC(Y, W, A, family='nb')
    assert isinstance(result, pd.DataFrame)
    assert 'trt' in result.columns
    assert len(result) == p * a
    assert result['tau'].notna().all()
    assert result['padj'].notna().all()