import pytest
import numpy as np

from ssr import rough_bergomi
from ssr.rough_bergomi import (
    RoughBergomiModel,
    _rough_bergomi_integrals_from_normal_jit,
)


def test_constructor_preserves_s0():
    model = RoughBergomiModel(
        params={"eta": 1.0, "H": 0.2, "rho": -0.5},
        xi0=lambda t: 0.04 * np.ones_like(t),
        s0=2.0,
    )

    assert model.s0 == 2.0


@pytest.mark.parametrize("Ts", [np.linspace(1e-3, 5.0, 30)])
@pytest.mark.parametrize("eta", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("rho", [-0.9, -0.5, -0.2])
@pytest.mark.parametrize("xi0", [lambda t: 0.04 * np.ones_like(t)])
def test_bg_beta_skew(Ts, eta, H, rho, xi0):
    """
    Check closed-form formulas for beta_1, beta_2, atm_skew_1, and atm_skew_2 when the
    initial forward variance curve is flat.
    """
    params = {"eta": eta, "H": H, "rho": rho}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)

    n_quad = 60
    out_bg = rbergomi.ssr_bg_all(Ts, n_quad=n_quad)

    assert np.allclose(out_bg["beta_1"], rbergomi._beta_1_bg_flat(Ts))
    assert np.allclose(out_bg["beta_2"], rbergomi._beta_2_bg_flat(Ts))
    assert np.allclose(out_bg["atm_skew_1"], rbergomi._atm_skew_1_bg_flat(Ts))
    assert np.allclose(out_bg["atm_skew_2"], rbergomi._atm_skew_2_bg_flat(Ts))


def test_xi0_shifted_h_half_uses_bergomi_factor():
    xi0 = lambda t: 0.04 * np.exp(-0.1 * t)
    params = {"eta": 1.0, "H": 0.5, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)

    u = np.array([0.0, 0.25, 1.0])
    eps = 1e-3
    expected = xi0(u) * np.exp(eps * params["eta"] * params["rho"] / np.sqrt(0.04))

    assert np.allclose(rbergomi.xi0_shifted(u, eps), expected)


def test_xi0_shifted_h_not_half_uses_deterministic_factor():
    xi0 = lambda t: 0.04 * np.exp(-0.1 * t)
    params = {"eta": 1.0, "H": 0.3, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)

    u = np.array([0.0, 0.25, 1.0])
    eps = 1e-3
    c = eps * params["eta"] * params["rho"] * np.sqrt(2.0 * params["H"])
    c /= np.sqrt(0.04)
    expected = xi0(u)
    expected[1:] *= 1.0 + c * u[1:] ** (params["H"] - 0.5)

    assert np.allclose(rbergomi.xi0_shifted(u, eps), expected)


def test_c_x_xi_gauss_matches_scipy_quad():
    xi0 = lambda t: 0.04 + 0.01 * np.asarray(t)
    params = {"eta": 1.0, "H": 0.3, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)
    Ts = np.array([0.2, 0.5])

    assert np.allclose(
        rbergomi._c_x_xi_bg_gauss_quad(Ts, n_quad=25),
        rbergomi._c_x_xi_bg_scipy_quad(Ts),
        rtol=5e-6,
        atol=1e-9,
    )


def test_simulate_mc_h_half_scales_shifted_ssr_integrals():
    xi0 = lambda t: 0.04 * np.exp(-0.1 * t)
    params = {"eta": 1.0, "H": 0.5, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)

    eps = 1e-3
    out = rbergomi.simulate_mc(
        tab_t=np.linspace(0.0, 1.0, 8),
        n_mc=12,
        n_loop=3,
        seed=1,
        eps_ssr=eps,
    )
    shift_factor = np.exp(eps * params["eta"] * params["rho"] / np.sqrt(0.04))

    assert np.allclose(out["int_v_dt_shifted"], shift_factor * out["int_v_dt"])
    assert np.allclose(
        out["int_sqrt_v_dw_shifted"],
        np.sqrt(shift_factor) * out["int_sqrt_v_dw"],
    )


def test_simulate_mc_reuses_cholesky_for_same_grid(monkeypatch):
    xi0 = lambda t: 0.04 * np.ones_like(t)
    params = {"eta": 1.0, "H": 0.3, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)
    tab_t = np.linspace(0.0, 1.0, 6)
    calls = 0
    original = rbergomi.cholesky_cov_matrix

    def wrapped_cholesky_cov_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rbergomi, "cholesky_cov_matrix", wrapped_cholesky_cov_matrix)

    first = rbergomi.simulate_mc(tab_t=tab_t, n_mc=8, n_loop=2, seed=1)
    second = rbergomi.simulate_mc(tab_t=tab_t, n_mc=8, n_loop=2, seed=1)

    assert calls == 1
    assert np.allclose(first["int_v_dt"], second["int_v_dt"])
    assert np.allclose(first["int_sqrt_v_dw"], second["int_sqrt_v_dw"])


def test_simulate_mc_reuses_unit_cholesky_for_uniform_grids(monkeypatch):
    xi0 = lambda t: 0.04 * np.ones_like(t)
    params = {"eta": 1.0, "H": 0.3, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)
    calls = 0
    original = rbergomi.cholesky_cov_matrix

    def wrapped_cholesky_cov_matrix(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rbergomi, "cholesky_cov_matrix", wrapped_cholesky_cov_matrix)

    rbergomi.simulate_mc(tab_t=np.linspace(0.0, 0.5, 6), n_mc=8, n_loop=2, seed=1)
    rbergomi.simulate_mc(tab_t=np.linspace(0.0, 1.2, 6), n_mc=8, n_loop=2, seed=2)

    assert calls == 1


def test_scaled_unit_cholesky_matches_full_covariance():
    xi0 = lambda t: 0.04 * np.ones_like(t)
    params = {"eta": 1.0, "H": 0.3, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)
    tab_t = np.linspace(0.0, 0.7, 6)

    chol = rbergomi._cached_cholesky_cov_matrix(tab_t, conditioning=True)
    cov = rbergomi.cholesky_cov_matrix(tab_t, conditioning=True, return_cov=True)

    assert np.allclose(chol @ chol.T, cov)


def test_simulate_mc_uses_triangular_blas_for_rough_case(monkeypatch):
    xi0 = lambda t: 0.04 * np.ones_like(t)
    params = {"eta": 1.0, "H": 0.3, "rho": -0.5}
    rbergomi = RoughBergomiModel(params=params, xi0=xi0)
    calls = 0
    original = rough_bergomi.blas.dtrmm

    def wrapped_dtrmm(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(rough_bergomi.blas, "dtrmm", wrapped_dtrmm)

    rbergomi.simulate_mc(tab_t=np.linspace(0.0, 1.0, 6), n_mc=8, n_loop=2, seed=1)

    assert calls == 2


def test_rough_bergomi_integrals_jit_matches_vectorized_reduction():
    n_disc = 5
    n_mc = 4
    eta = 1.1
    dt = 0.2
    tab_t = np.linspace(0.0, 1.0, n_disc + 1)
    xi0_t = 0.04 + 0.01 * tab_t
    drift_t = -0.5 * eta**2 * tab_t**0.6
    shift_t = 1.0 + 0.03 * np.sqrt(tab_t)
    rng = np.random.default_rng(123)
    normal = rng.normal(size=(2 * n_disc, n_mc))

    kappa = 0.5 + 0.1 * tab_t[:-1]

    out = _rough_bergomi_integrals_from_normal_jit(
        normal, xi0_t, drift_t, shift_t, dt, eta, True, kappa, True
    )

    y = np.empty((n_disc + 1, n_mc))
    y[0, :] = 0.0
    y[1:, :] = normal[:n_disc, :]
    w = normal[n_disc:, :]
    dw = np.empty_like(w)
    dw[0, :] = w[0, :]
    dw[1:, :] = w[1:, :] - w[:-1, :]

    exp_term = np.exp(eta * y + drift_t[:, np.newaxis])
    v_prev = xi0_t[:-1, np.newaxis] * exp_term[:-1, :]
    v_next = xi0_t[1:, np.newaxis] * exp_term[1:, :]
    sqrt_v_prev = np.sqrt(v_prev)

    expected_int_v_dt = 0.5 * dt * np.sum(v_prev + v_next, axis=0)
    expected_int_sqrt_v_dw = np.sum(sqrt_v_prev * dw, axis=0)
    expected_int_v_dt_shifted = (
        0.5
        * dt
        * np.sum(
            shift_t[:-1, np.newaxis] * v_prev + shift_t[1:, np.newaxis] * v_next,
            axis=0,
        )
    )
    expected_int_sqrt_v_dw_shifted = np.sum(
        np.sqrt(shift_t[:-1, np.newaxis]) * sqrt_v_prev * dw,
        axis=0,
    )

    expected_int_sqrt_v_k_dw = np.sum(
        sqrt_v_prev[1:] * kappa[:-1, np.newaxis] * dw[1:], axis=0
    )
    expected_int_v_k_dt = dt * np.sum(v_next * kappa[:, np.newaxis], axis=0)

    assert np.allclose(out[0], expected_int_v_dt)
    assert np.allclose(out[1], expected_int_sqrt_v_dw)
    assert np.allclose(out[2], expected_int_v_dt_shifted)
    assert np.allclose(out[3], expected_int_sqrt_v_dw_shifted)
    assert np.allclose(out[4], expected_int_sqrt_v_k_dw)
    assert np.allclose(out[5], expected_int_v_k_dt)


def test_fukasawa_kernel_right_point_matches_kernel():
    model = RoughBergomiModel(
        params={"eta": 1.3, "H": 0.15, "rho": -0.6},
        xi0=lambda t: 0.04 * np.ones_like(t),
    )
    tab_t = np.array([0.0, 0.1, 0.4, 1.0])

    def k(s):
        return model.rho * model.eta * np.sqrt(2.0 * model.H) * s ** (model.H - 0.5)

    expected = k(tab_t[1:])
    actual = model._fukasawa_kernel_right_point(tab_t)
    assert np.allclose(actual, expected)


def test_ssr_fukasawa_close_to_finite_difference_ssr():
    model = RoughBergomiModel(
        params={"eta": 1.9, "H": 0.3, "rho": -0.9},
        xi0=lambda t: 0.235**2 * np.ones_like(t),
    )
    T = np.array([0.5, 1.0])
    out_fd = model.ssr_all(
        T=T, n_mc=100_000, n_disc=200, eps_ssr=1e-3, n_quad=20, seed=1
    )
    out_fukasawa = model.ssr_fukasawa_all(T=T, n_mc=100_000, n_disc=200, seed=1)

    assert np.allclose(out_fukasawa["ssr_fukasawa"], out_fd["ssr_fd"], atol=0.15)
