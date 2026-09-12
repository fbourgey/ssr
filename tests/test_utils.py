import numpy as np
import pandas as pd
import pytest
from scipy.special import roots_jacobi

from ssr.utils import black_impvol, black_otm_impvol_mc, black_price
from ssr.utils import cholesky_from_svd
from ssr.utils import gauss_hermite, gauss_legendre
from ssr.utils import implied_vol_from_paths
from ssr.utils import jacobi_quadrature
from ssr.utils import lewis_formula_otm_price, mittag_leffler_two
from ssr.utils import non_uniform_grid, plot_ivols_mc


def test_black_impvol_recovers_vector_vols():
    K = np.array([0.8, 1.0, 1.2])
    T = 1.5
    F = 1.0
    vol = np.array([0.15, 0.2, 0.35])
    opttype = np.array([-1.0, 1.0, 1.0])

    value = black_price(K=K, T=T, F=F, vol=vol, opttype=opttype)
    impvol = black_impvol(K=K, T=T, F=F, value=value, opttype=opttype, TOL=1e-10)

    assert np.allclose(impvol, vol, atol=1e-8)


@pytest.mark.parametrize("shape", [(2, 3), (2, 2, 3)])
@pytest.mark.parametrize("opttype", [1, np.array([-1, 1, 1])])
def test_black_impvol_recovers_multidimensional_vols(shape, opttype):
    K = np.linspace(0.8, 1.2, np.prod(shape)).reshape(shape)
    vol = np.linspace(0.15, 0.35, K.size).reshape(shape)
    value = black_price(K=K, T=1.5, F=1.0, vol=vol, opttype=opttype)

    actual = black_impvol(K=K, T=1.5, F=1.0, value=value, opttype=opttype)

    assert actual.shape == shape
    np.testing.assert_allclose(actual, vol, atol=1e-8)


def test_black_impvol_multidimensional_invalid_prices_preserve_positions():
    K = np.array([[0.9, 1.0], [1.1, 1.2]])
    value = black_price(K=K, T=1.0, F=1.0, vol=0.2)
    value[0, 1] = np.nan
    value[1, 0] = 2.0

    actual = black_impvol(K=K, T=1.0, F=1.0, value=value)

    np.testing.assert_allclose(actual, [[0.2, np.nan], [np.nan, 0.2]], atol=1e-8)


def test_black_price_handles_zero_time_and_invalid_inputs():
    price = black_price(
        K=np.array([0.9, 1.1, -1.0]),
        T=0.0,
        F=1.0,
        vol=0.2,
        opttype=np.array([1.0, -1.0, 1.0]),
    )

    assert np.allclose(price[:2], [0.1, 0.1])
    assert np.isnan(price[2])

    with pytest.raises(ValueError, match="opttype"):
        black_price(K=1.0, T=1.0, F=1.0, vol=0.2, opttype=0.0)


def test_black_impvol_rejects_bad_opttype_shape():
    K = np.array([0.9, 1.1])
    value = np.array([0.1, 0.1])

    with pytest.raises(ValueError, match="opttype"):
        black_impvol(K=K, T=1.0, F=1.0, value=value, opttype=np.ones(3))


def test_black_impvol_returns_nan_for_price_above_max_vol_bound():
    K = np.array([1.0])

    impvol = black_impvol(K=K, T=1.0, F=1.0, value=np.array([2.0]))

    assert np.isnan(impvol[0])


def test_black_otm_impvol_mc_validates_paths_and_returns_vector():
    S = np.array([0.8, 0.95, 1.05, 1.2])

    impvol = black_otm_impvol_mc(S=S, k=np.array([-0.05, 0.05]), T=1.0)

    assert impvol.shape == (2,)

    with pytest.raises(ValueError, match="positive simulated prices"):
        black_otm_impvol_mc(S=np.array([1.0, 0.0]), k=0.0, T=1.0)


def test_lewis_formula_rejects_nonpositive_maturity():
    with pytest.raises(ValueError, match="T must be positive"):
        lewis_formula_otm_price(lambda u, T: np.ones_like(T), k=0.0, T=0.0)


def test_mittag_leffler_two_rejects_nonpositive_parameters():
    with pytest.raises(ValueError, match="alpha and beta"):
        mittag_leffler_two(0.0, alpha=0.0, beta=1.0)


def test_implied_vol_from_paths_returns_vector_skew():
    int_v_dt = np.full(6, 0.04)
    int_sqrt_v_dw = np.array([-0.3, -0.15, -0.02, 0.06, 0.18, 0.31])

    impvol, skew = implied_vol_from_paths(
        k=np.array([-0.1, 0.0, 0.1]),
        T=1.0,
        int_v_dt=int_v_dt,
        int_sqrt_v_dw=int_sqrt_v_dw,
        s0=1.0,
        return_skew=True,
    )

    assert impvol.shape == (3,)
    assert skew.shape == (3,)
    assert np.all(np.isfinite(impvol))
    assert np.all(np.isfinite(skew))


def test_implied_vol_from_paths_conditioning_matches_scalar_loop():
    k = np.array([-0.1, 0.0, 0.1])
    T = 1.0
    int_v_dt = np.array([0.01, 0.025, 0.04, 0.055])
    int_sqrt_v_dw = np.array([-0.2, -0.05, 0.08, 0.21])
    s0 = 1.0
    rho_cond = -0.5

    impvol, skew = implied_vol_from_paths(
        k=k,
        T=T,
        int_v_dt=int_v_dt,
        int_sqrt_v_dw=int_sqrt_v_dw,
        s0=s0,
        conditioning=True,
        return_skew=True,
        rho_cond=rho_cond,
    )

    scalar_impvol = []
    scalar_skew = []
    for k_i in k:
        impvol_i, skew_i = implied_vol_from_paths(
            k=k_i,
            T=T,
            int_v_dt=int_v_dt,
            int_sqrt_v_dw=int_sqrt_v_dw,
            s0=s0,
            conditioning=True,
            return_skew=True,
            rho_cond=rho_cond,
        )
        scalar_impvol.append(impvol_i.item())
        scalar_skew.append(skew_i.item())

    assert np.allclose(impvol, scalar_impvol)
    assert np.allclose(skew, scalar_skew)


def test_gauss_hermite_matches_probabilist_scaling_and_is_cache_safe():
    knots, weights = gauss_hermite(4)
    raw_knots, raw_weights = np.polynomial.hermite.hermgauss(4)

    assert np.allclose(knots, raw_knots * np.sqrt(2.0))
    assert np.allclose(weights, raw_weights / np.sqrt(np.pi))

    knots[0] = 999.0
    weights[0] = 999.0
    fresh_knots, fresh_weights = gauss_hermite(4)

    assert fresh_knots[0] != 999.0
    assert fresh_weights[0] != 999.0


def test_gauss_legendre_matches_interval_scaling_and_is_cache_safe():
    knots, weights = gauss_legendre(2.0, 5.0, 4)
    raw_knots, raw_weights = np.polynomial.legendre.leggauss(4)

    assert np.allclose(knots, 1.5 * raw_knots + 3.5)
    assert np.allclose(weights, 1.5 * raw_weights)

    knots[0] = 999.0
    weights[0] = 999.0
    fresh_knots, fresh_weights = gauss_legendre(2.0, 5.0, 4)

    assert fresh_knots[0] != 999.0
    assert fresh_weights[0] != 999.0


def test_jacobi_quadrature_matches_scipy_and_is_cache_safe():
    knots, weights = jacobi_quadrature(4, alpha=0.2, beta=0.7)
    raw_knots, raw_weights = roots_jacobi(4, 0.2, 0.7)

    assert np.allclose(knots, raw_knots)
    assert np.allclose(weights, raw_weights)

    knots[0] = 999.0
    weights[0] = 999.0
    fresh_knots, fresh_weights = jacobi_quadrature(4, alpha=0.2, beta=0.7)

    assert fresh_knots[0] != 999.0
    assert fresh_weights[0] != 999.0


@pytest.mark.parametrize(
    "func",
    [
        gauss_hermite,
        lambda n: gauss_legendre(0.0, 1.0, n),
        lambda n: jacobi_quadrature(n, alpha=0.2, beta=0.7),
    ],
)
@pytest.mark.parametrize("n", [0, -1, True])
def test_gauss_quadrature_rejects_invalid_order(func, n):
    with pytest.raises(ValueError, match="positive integer"):
        func(n)


def test_non_uniform_grid_keeps_power_keyword():
    grid = non_uniform_grid(2.0, 10.0, 3, power=2.0)

    assert np.allclose(grid, [2.0, 2.88888889, 5.55555556, 10.0])


def test_cholesky_from_svd_returns_psd_factor():
    a = np.array([[1.0, 0.5], [0.5, 0.25]])

    factor = cholesky_from_svd(a)

    assert np.allclose(factor @ factor.T, a)


def test_plot_ivols_mc_uses_output_indices_for_sliced_expiries():
    ivol_data = pd.DataFrame(
        {
            "Texp": [0.5, 0.5, 0.5, 1.0, 1.0, 1.0],
            "Bid": [0.18, 0.2, 0.22, 0.19, 0.21, 0.23],
            "Ask": [0.2, 0.22, 0.24, 0.21, 0.23, 0.25],
            "Fwd": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "Strike": [0.9, 1.0, 1.1, 0.9, 1.0, 1.1],
        }
    )
    mc_matrix = np.array(
        [
            [0.8, 0.95, 1.05, 1.2],
            [0.75, 0.9, 1.1, 1.25],
        ]
    )

    result = plot_ivols_mc(ivol_data, slices=[1], mc_matrix=mc_matrix, plot=False)

    assert np.allclose(result["expiries"], [1.0])
    assert result["atm_vols"].shape == (1,)
    assert result["atm_vols_mc"].shape == (1,)
    assert np.isfinite(result["atm_vols"][0])
