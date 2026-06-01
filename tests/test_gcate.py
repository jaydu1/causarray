import numpy as np
import pytest
from unittest.mock import patch
from causarray.gcate import fit_gcate
import causarray.gcate_glm as gcate_glm

def test_fit_gcate():
    # Sample input data
    Y = np.random.randint(0, 10, (100, 10))
    X = np.random.randn(100, 4)
    A = np.random.randn(100, 1)
    r = 2
    family = 'nb'
    disp_glm = None
    disp_family = 'poisson'
    offset = np.ones(100)
    kwargs_ls_1 = {}
    kwargs_ls_2 = {}
    kwargs_es_1 = {}
    kwargs_es_2 = {}
    c1 = 0.1

    # Call the function
    res_1, res_2 = fit_gcate(Y, X, A, r, family, disp_glm, disp_family, offset,
                             kwargs_ls_1, kwargs_ls_2, kwargs_es_1, kwargs_es_2,
                             c1)

    # Assert the output shapes and types
    assert isinstance(res_1, dict)
    assert isinstance(res_2, dict)
    assert res_1['X_U'].shape[1] == X.shape[1] + A.shape[1] + r
    assert res_2['X_U'].shape[1] == X.shape[1] + A.shape[1] + r

def test_fit_gcate_invalid_input():
    # Invalid input: Y is not a 2D array
    Y_invalid = np.random.randint(0, 10, (100,))
    X = np.random.randn(100, 4)
    A = np.random.randn(100, 1)
    r = 2
    family = 'nb'
    disp_glm = None
    disp_family = 'poisson'
    offset = np.ones(100)
    kwargs_ls_1 = {}
    kwargs_ls_2 = {}
    kwargs_es_1 = {}
    kwargs_es_2 = {}
    c1 = 0.1

    with pytest.raises(ValueError):
        fit_gcate(Y_invalid, X, A, r, family, disp_glm, disp_family, offset,
                  kwargs_ls_1, kwargs_ls_2, kwargs_es_1, kwargs_es_2, c1)

    # Invalid input: X is not a 2D array
    Y = np.random.randint(0, 10, (100, 10))
    X_invalid = np.random.randn(100)
    
    with pytest.raises(ValueError):
        fit_gcate(Y, X_invalid, A, r, family, disp_glm, disp_family, offset,
                  kwargs_ls_1, kwargs_ls_2, kwargs_es_1, kwargs_es_2, c1)

def test_fit_gcate_non_integer_Y():
    Y = np.random.rand(100, 10) * 10
    X = np.random.randn(100, 4)
    A = np.random.randn(100, 1)
    r = 2
    family = 'nb'
    disp_glm = None
    disp_family = 'poisson'
    offset = np.ones(100)
    kwargs_ls_1 = {}
    kwargs_ls_2 = {}
    kwargs_es_1 = {}
    kwargs_es_2 = {}
    c1 = 0.1

    # Call the function
    res_1, res_2 = fit_gcate(Y, X, A, r, family, disp_glm, disp_family, offset,
                             kwargs_ls_1, kwargs_ls_2, kwargs_es_1, kwargs_es_2,
                             c1)

    # Assert the output shapes and types
    assert isinstance(res_1, dict)
    assert isinstance(res_2, dict)
    assert res_1['X_U'].shape[1] == X.shape[1] + A.shape[1] + r
    assert res_2['X_U'].shape[1] == X.shape[1] + A.shape[1] + r

def test_fit_gcate_near_singular_X():
    Y = np.random.randint(0, 10, (100, 10))
    X = np.ones((100, 4))
    A = np.ones((100, 1))
    r = 2

    with pytest.raises(ValueError):
        fit_gcate(Y, X, A, r)


# ---------------------------------------------------------------------------
# Backend integration via fit_gcate
# ---------------------------------------------------------------------------

class TestBackendIntegration:
    def test_backend_toggle_propagates_through_fit_gcate(self):
        """_USE_FAST_BACKEND=False must cause fit_gcate to use statsmodels path."""
        np.random.seed(0)
        n, p = 60, 20
        Y = np.random.negative_binomial(5, 0.5, (n, p))
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        A = np.random.binomial(1, 0.3, (n, 1)).astype(float)

        with gcate_glm._backend_override("original"):
            with patch.object(gcate_glm, "fit_glm", wraps=gcate_glm.fit_glm) as mock_fg:
                fit_gcate(Y, X, A, r=1, family="nb", offset=False, backend="original")
                assert mock_fg.called, (
                    "fit_glm (statsmodels) was not called despite backend='original'"
                )

    def test_backend_original_param_accepted(self):
        """fit_gcate(..., backend='original') must complete and return dicts."""
        np.random.seed(1)
        n, p = 60, 20
        Y = np.random.negative_binomial(5, 0.5, (n, p))
        X = np.column_stack([np.ones(n), np.random.randn(n)])
        A = np.random.binomial(1, 0.3, (n, 1)).astype(float)

        res_1, res_2 = fit_gcate(Y, X, A, r=1, family="nb", offset=False, backend="original")
        assert isinstance(res_1, dict)
        assert isinstance(res_2, dict)