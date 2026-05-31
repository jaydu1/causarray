import numpy as np
import statsmodels.api as sm
from causarray.nb_glm_fast import fit_glm_fast


def test_fit_glm_fast_poisson_coefficients():
    """fit_glm_fast (Poisson) coefficients agree with statsmodels within 20%."""
    np.random.seed(0)
    n, p = 120, 8
    X = np.column_stack([np.ones(n), np.random.normal(size=n)])
    B_true = np.random.normal(0, 0.3, (p, 2))
    Y = np.random.poisson(np.exp(X @ B_true.T)).astype(float)

    B, Yhat, disp_glm, offsets, resid_dev = fit_glm_fast(Y, X, family="poisson")

    assert B.shape == (p, 2), f"expected ({p}, 2), got {B.shape}"
    assert Yhat.shape == (n, p)
    assert disp_glm is None  # Poisson family has no dispersion parameter
    assert resid_dev.shape == (n, p)
    assert np.all(np.isfinite(B))

    # Compare against statsmodels gene-by-gene
    for j in range(p):
        mod = sm.GLM(Y[:, j], X, family=sm.families.Poisson()).fit()
        np.testing.assert_allclose(B[j], mod.params, rtol=0.2,
                                   err_msg=f"gene {j} coefficient mismatch")


def test_fit_glm_fast_output_shapes_nb():
    """fit_glm_fast (NB) returns arrays with correct shapes."""
    np.random.seed(1)
    n, p = 80, 12
    X = np.ones((n, 1))
    Y = np.random.negative_binomial(5, 0.5, size=(n, p)).astype(float)

    B, Yhat, disp_glm, offsets, resid_dev = fit_glm_fast(Y, X, family="nb")

    assert B.shape == (p, 1)
    assert Yhat.shape == (n, p)
    assert disp_glm.shape == (p,)
    assert np.all(np.isfinite(Yhat))
    assert np.all(Yhat >= 0)


def test_fit_glm_fast_with_treatment():
    """fit_glm_fast with A treatment indicator returns (p, d+1) coefficients."""
    np.random.seed(2)
    n, p = 100, 10
    X = np.ones((n, 1))
    A = np.zeros((n, 1)); A[:40] = 1.0
    Y = np.random.poisson(3, size=(n, p)).astype(float)

    B, Yhat, disp_glm, offsets, resid_dev = fit_glm_fast(Y, X, A=A, family="poisson")

    assert B.shape == (p, 2), f"expected ({p}, 2), got {B.shape}"
    assert np.all(np.isfinite(B))


if __name__ == "__main__":
    test_fit_glm_fast_poisson_coefficients()
    test_fit_glm_fast_output_shapes_nb()
    test_fit_glm_fast_with_treatment()
    print("All tests passed.")

