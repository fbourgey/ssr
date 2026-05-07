import numpy as np
import pytest

from heston import HestonModel


def test_bg_closed_forms_match_gauss_quadrature():
    params = {"kappa": 1.3, "rho": -0.7, "eta": 0.8, "vbar": 0.05, "v": 0.04}
    heston = HestonModel(params=params)
    Ts = np.array([0.1, 0.7, 2.0])
    n_quad = 80

    assert np.allclose(heston.c_x_xi_bg(Ts), heston.c_x_xi_bg(Ts, n_quad=n_quad))
    assert np.allclose(heston.c_mu_bg(Ts), heston.c_mu_bg(Ts, n_quad=n_quad))


def test_bg_zero_kappa_limits_match_gauss_quadrature():
    params = {"kappa": 0.0, "rho": -0.7, "eta": 0.8, "vbar": 0.05, "v": 0.04}
    heston = HestonModel(params=params)
    Ts = np.array([0.1, 0.7, 2.0])
    n_quad = 80

    assert np.allclose(heston.c_x_xi_bg(Ts), heston.c_x_xi_bg(Ts, n_quad=n_quad))
    assert np.allclose(heston.c_mu_bg(Ts), heston.c_mu_bg(Ts, n_quad=n_quad))
    assert np.allclose(
        heston._dxi_c_x_xi_beta_2(Ts),
        (heston.rho * heston.eta) ** 2 * Ts**2 / 2.0,
    )


def test_bg_zero_kappa_limits_match_small_positive_kappa():
    params = {"kappa": 0.0, "rho": -0.7, "eta": 0.8, "vbar": 0.05, "v": 0.04}
    heston_zero = HestonModel(params=params)
    heston_small = HestonModel(params=params | {"kappa": 1e-8})
    Ts = np.array([0.1, 0.7, 2.0])

    assert np.allclose(
        heston_zero.c_x_xi_bg(Ts),
        heston_small.c_x_xi_bg(Ts),
        rtol=1e-6,
        atol=1e-10,
    )
    assert np.allclose(
        heston_zero.c_mu_bg(Ts),
        heston_small.c_mu_bg(Ts),
        rtol=1e-6,
        atol=1e-10,
    )
    assert np.allclose(
        heston_zero._dxi_c_x_xi_beta_2(Ts),
        heston_small._dxi_c_x_xi_beta_2(Ts),
        rtol=1e-6,
        atol=1e-10,
    )


def test_c_x_xi_gauss_matches_scipy_quad():
    params = {"kappa": 1.3, "rho": -0.7, "eta": 0.8, "vbar": 0.05, "v": 0.04}
    heston = HestonModel(params=params)
    Ts = np.array([0.2, 0.5, 1.0])

    assert np.allclose(
        heston._c_x_xi_bg_gauss_quad(Ts, n_quad=40),
        heston._c_x_xi_bg_scipy_quad(Ts),
        rtol=1e-10,
        atol=1e-14,
    )


def test_ssr_bg_all_first_order_correction_includes_vs_term():
    params = {"kappa": 1.3, "rho": -0.7, "eta": 0.8, "vbar": 0.05, "v": 0.04}
    heston = HestonModel(params=params)
    Ts = np.array([0.1, 0.7, 2.0])

    out = heston.ssr_bg_all(Ts)

    assert np.allclose(
        out["ssr_1"],
        out["ssr_vs_1"] + out["beta_2"] / out["atm_skew_1"],
    )


@pytest.mark.parametrize(
    ("eps", "rtol"),
    [
        (1e-3, 4e-3),
        (1e-4, 5e-4),
        (1e-5, 1e-4),
    ],
)
def test_ssr_characteristic_function_matches_finite_difference(eps, rtol):
    params = {"kappa": 1.3, "rho": -0.7, "eta": 0.8, "vbar": 0.05, "v": 0.04}
    heston = HestonModel(params=params)
    Ts = np.array([0.1, 0.7, 2.0])

    assert np.allclose(
        heston.ssr_cf(Ts),
        heston.ssr_fd(Ts, eps=eps),
        rtol=rtol,
        atol=1e-6,
    )


@pytest.mark.parametrize("eps", [1e-3, 1e-4, 1e-5])
def test_atm_skew_characteristic_function_matches_finite_difference(eps):
    params = {"kappa": 1.3, "rho": -0.7, "eta": 0.8, "vbar": 0.05, "v": 0.04}
    heston = HestonModel(params=params)
    Ts = np.array([0.1, 0.7, 2.0])

    assert np.allclose(
        heston.atm_skew_cf(Ts),
        heston.atm_skew_fd(Ts, eps=eps),
        rtol=5e-5,
        atol=1e-8,
    )
