import numpy as np
import pandas as pd
import pytest
from causarray.DR_learner import LFC


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(0)
    Y = rng.poisson(5, (100, 10)).astype(float)
    W = rng.standard_normal((100, 5))
    A = rng.binomial(1, 0.5, 100)
    Y += rng.poisson(1, (100, 10)) * A[:, None]
    return Y, W, A


class TestLFCOutputSchema:
    def test_output_columns(self, sample_data):
        Y, W, A = sample_data
        result, estimation = LFC(Y, W, A)
        assert isinstance(result, pd.DataFrame)
        for col in ('tau', 'std', 'stat', 'rej', 'pvalue', 'padj',
                    'pvalue_emp_null_adj', 'padj_emp_null_adj'):
            assert col in result.columns

    def test_with_offset(self, sample_data):
        Y, W, A = sample_data
        rng = np.random.default_rng(1)
        offset = np.log(rng.poisson(5, 100).clip(1))
        result, estimation = LFC(Y, W, A, offset=offset)
        assert isinstance(result, pd.DataFrame)
        assert 'tau' in result.columns

    def test_with_fdx(self, sample_data):
        Y, W, A = sample_data
        result, estimation = LFC(Y, W, A, fdx=True)
        assert isinstance(result, pd.DataFrame)
        assert 'tau' in result.columns

    def test_with_custom_family(self, sample_data):
        Y, W, A = sample_data
        result, estimation = LFC(Y, W, A, family='poisson')
        assert isinstance(result, pd.DataFrame)
        assert 'tau' in result.columns


class TestLFCMultiTreatment:
    def test_multi_treatment(self):
        """LFC with multiple perturbations exercises the per-perturbation fast path."""
        np.random.seed(42)
        n, p, a = 200, 50, 3
        W = np.random.normal(0, 1, (n, 3))
        A = np.zeros((n, a))
        A[50:100, 0] = 1
        A[100:150, 1] = 1
        A[150:200, 2] = 1
        Y = np.random.poisson(5, (n, p))
        Y[50:100] += np.random.poisson(2, (50, p))

        result, estimation = LFC(Y, W, A, family='nb')
        assert isinstance(result, pd.DataFrame)
        assert 'trt' in result.columns
        assert len(result) == p * a
        assert result['tau'].notna().all()
        assert result['padj'].notna().all()
