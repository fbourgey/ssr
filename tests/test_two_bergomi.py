import pytest
import numpy as np
from two_bergomi import (
    TwoFactorBergomiModel,
    _func_I,
    _ou_factor_paths,
    _two_bergomi_integrals_from_normal_jit,
    exp_decay_kernel_matrices,
)


FLAT_PARAMS = {
    "w": 2 * 1.74,
    "k1": 5.25,
    "k2": 0.28,
    "theta": 0.245,
    "rho12": 0.0,
    "rhoS1": -0.759,
    "rhoS2": -0.487,
}

NONFLAT_PARAMS = {
    "w": 1.0,
    "k1": 1.3,
    "k2": 2.1,
    "theta": 0.4,
    "rho12": 0.1,
    "rhoS1": -0.5,
    "rhoS2": -0.2,
}


@pytest.mark.parametrize("name", ["rho12", "rhoS1", "rhoS2"])
@pytest.mark.parametrize("value", [-1.0, 1.0])
def test_rejects_degenerate_correlation_boundaries(name, value):
    params = NONFLAT_PARAMS | {name: value}

    with pytest.raises(ValueError, match=rf"{name} must be in \(-1, 1\)"):
        TwoFactorBergomiModel(params=params, xi0=lambda t: 0.04 * np.ones_like(t))


@pytest.mark.parametrize("Ts", [np.linspace(1e-3, 5.0, 30)])
@pytest.mark.parametrize("xi0", [lambda t: 0.04 * np.ones_like(t)])
def test_bg_beta_skew(Ts, xi0):
    """
    Check closed-form formulas for beta_1, beta_2, atm_skew_1, and atm_skew_2 when the
    initial forward variance curve is flat.
    """
    two_bergomi = TwoFactorBergomiModel(params=FLAT_PARAMS, xi0=xi0)

    n_quad = 30
    out_bg = two_bergomi.ssr_bg_all(Ts, n_quad=n_quad)

    assert np.allclose(out_bg["beta_1"], two_bergomi._beta_1_bg_flat(Ts))
    assert np.allclose(out_bg["beta_2"], two_bergomi._beta_2_bg_flat(Ts))
    assert np.allclose(out_bg["atm_skew_1"], two_bergomi._atm_skew_1_bg_flat(Ts))
    assert np.allclose(out_bg["atm_skew_2"], two_bergomi._atm_skew_2_bg_flat(Ts))


def test_ssr_bg_all_uses_flat_closed_forms(monkeypatch):
    two_bergomi = TwoFactorBergomiModel(
        params=FLAT_PARAMS, xi0=lambda t: 0.04 * np.ones_like(t)
    )
    Ts = np.array([0.1, 0.7, 2.0])

    def fail_if_quadrature_path_is_used(*args, **kwargs):
        raise AssertionError("flat ssr_bg_all should use closed forms")

    monkeypatch.setattr(
        two_bergomi, "_c_x_xi_bg_gauss_quad", fail_if_quadrature_path_is_used
    )
    monkeypatch.setattr(
        two_bergomi, "_c_mu_bg_gauss_quad", fail_if_quadrature_path_is_used
    )
    monkeypatch.setattr(
        two_bergomi, "_beta_1_gauss_quad", fail_if_quadrature_path_is_used
    )
    monkeypatch.setattr(
        two_bergomi, "_beta_2_gauss_quad", fail_if_quadrature_path_is_used
    )

    out_bg = two_bergomi.ssr_bg_all(Ts, n_quad=30)

    assert np.allclose(out_bg["beta_1"], two_bergomi._beta_1_bg_flat(Ts))
    assert np.allclose(out_bg["beta_2"], two_bergomi._beta_2_bg_flat(Ts))
    assert np.allclose(out_bg["atm_skew_1"], two_bergomi._atm_skew_1_bg_flat(Ts))
    assert np.allclose(out_bg["atm_skew_2"], two_bergomi._atm_skew_2_bg_flat(Ts))


def test_flat_bg_components_skip_quadrature(monkeypatch):
    two_bergomi = TwoFactorBergomiModel(
        params=FLAT_PARAMS, xi0=lambda t: 0.04 * np.ones_like(t)
    )
    Ts = np.array([0.1, 0.7, 2.0])

    def fail_if_quadrature_is_used(*args, **kwargs):
        raise AssertionError("flat components should use closed forms")

    monkeypatch.setattr("two_bergomi.gauss_legendre", fail_if_quadrature_is_used)

    assert np.allclose(
        two_bergomi._beta_1_gauss_quad(Ts, n_quad=30),
        two_bergomi._beta_1_bg_flat(Ts),
    )
    assert np.allclose(
        two_bergomi._c_x_xi_bg_gauss_quad(Ts, n_quad=30),
        two_bergomi._c_x_xi_bg_flat(Ts),
    )
    assert np.allclose(
        two_bergomi._c_mu_bg_gauss_quad(Ts, n_quad=30),
        two_bergomi._c_mu_bg_flat(Ts),
    )


def test_ssr_bg_all_nonflat_reuses_first_order_terms(monkeypatch):
    two_bergomi = TwoFactorBergomiModel(
        params=NONFLAT_PARAMS, xi0=lambda t: 0.04 + 0.01 * np.asarray(t)
    )
    calls = {"sig_varswap": 0, "c_x_xi": 0, "beta_1": 0}

    def sig_varswap(T, n_quad):
        calls["sig_varswap"] += 1
        return np.ones(2)

    def c_x_xi(T, n_quad):
        calls["c_x_xi"] += 1
        return np.array([0.02, 0.03])

    def beta_1(T, n_quad, sig_varswap):
        calls["beta_1"] += 1
        assert np.allclose(sig_varswap, np.ones(2))
        return np.array([0.2, 0.3])

    monkeypatch.setattr(two_bergomi, "sig_varswap", sig_varswap)
    monkeypatch.setattr(two_bergomi, "_c_x_xi_bg_gauss_quad", c_x_xi)
    monkeypatch.setattr(two_bergomi, "_beta_1_gauss_quad_from_sig_varswap", beta_1)
    monkeypatch.setattr(
        two_bergomi,
        "_dxi_c_x_xi_beta_2_gauss_quad",
        lambda T, n_quad: np.array([0.4, 0.6]),
    )
    monkeypatch.setattr(
        two_bergomi, "_c_mu_bg_gauss_quad", lambda T, n_quad: np.array([0.05, 0.07])
    )

    out_bg = two_bergomi.ssr_bg_all(np.array([0.5, 1.0]), n_quad=3)

    assert calls == {"sig_varswap": 1, "c_x_xi": 1, "beta_1": 1}
    assert np.allclose(
        out_bg["beta_2"],
        np.array([0.4 - 0.02 * 0.2, 0.6 - 0.03 * 0.3]) / np.array([2.0, 4.0]),
    )


def test_c_x_xi_explicit_matches_generic_gauss_legendre_for_nonflat_curve():
    two_bergomi = TwoFactorBergomiModel(
        params=NONFLAT_PARAMS, xi0=lambda t: 0.04 + 0.01 * np.asarray(t)
    )
    Ts = np.array([0.1, 0.7, 2.0])

    assert np.allclose(
        two_bergomi._c_x_xi_bg_gauss_quad_explicit(Ts, n_quad=20),
        two_bergomi._c_x_xi_bg_gauss_leg_quad(Ts, n_quad=20),
    )


def test_c_x_xi_gauss_matches_scipy_quad_for_nonflat_curve():
    two_bergomi = TwoFactorBergomiModel(
        params=NONFLAT_PARAMS, xi0=lambda t: 0.04 + 0.01 * np.asarray(t)
    )
    Ts = np.array([0.2, 0.5])

    assert np.allclose(
        two_bergomi._c_x_xi_bg_gauss_quad(Ts, n_quad=10),
        two_bergomi._c_x_xi_bg_scipy_quad(Ts),
        rtol=1e-10,
        atol=1e-14,
    )


def test_c_x_xi_scipy_gauss_and_flat_match_for_flat_curve():
    two_bergomi = TwoFactorBergomiModel(
        params=FLAT_PARAMS, xi0=lambda t: 0.04 * np.ones_like(t)
    )
    Ts = np.array([0.2, 0.5])

    expected = two_bergomi._c_x_xi_bg_flat(Ts)

    assert np.allclose(
        two_bergomi._c_x_xi_bg_scipy_quad(Ts),
        expected,
        rtol=1e-10,
        atol=1e-14,
    )
    assert np.allclose(
        two_bergomi._c_x_xi_bg_gauss_quad(Ts, n_quad=10),
        expected,
        rtol=1e-10,
        atol=1e-14,
    )


def test_c_mu_gauss_matches_scipy_quad_for_nonflat_curve():
    two_bergomi = TwoFactorBergomiModel(
        params=NONFLAT_PARAMS, xi0=lambda t: 0.04 + 0.01 * np.asarray(t)
    )
    Ts = np.array([0.2, 0.5])

    assert np.allclose(
        two_bergomi._c_mu_bg_gauss_quad(Ts, n_quad=10),
        two_bergomi._c_mu_bg_scipy_quad(Ts),
        rtol=1e-10,
        atol=1e-14,
    )


def test_c_mu_scipy_gauss_and_flat_match_for_flat_curve():
    two_bergomi = TwoFactorBergomiModel(
        params=FLAT_PARAMS, xi0=lambda t: 0.04 * np.ones_like(t)
    )
    Ts = np.array([0.2, 0.5])

    expected = two_bergomi._c_mu_bg_flat(Ts)

    assert np.allclose(
        two_bergomi._c_mu_bg_scipy_quad(Ts),
        expected,
        rtol=1e-10,
        atol=1e-14,
    )
    assert np.allclose(
        two_bergomi._c_mu_bg_gauss_quad(Ts, n_quad=10),
        expected,
        rtol=1e-10,
        atol=1e-14,
    )


def test_c_mu_flat_alt_matches_c_mu_flat():
    two_bergomi = TwoFactorBergomiModel(
        params=FLAT_PARAMS, xi0=lambda t: 0.04 * np.ones_like(t)
    )
    Ts = np.array([0.2, 0.5])

    assert np.allclose(
        two_bergomi._c_mu_bg_flat_alt(Ts),
        two_bergomi._c_mu_bg_flat(Ts),
        rtol=1e-10,
        atol=1e-14,
    )


def test_ou_factor_paths_matches_dense_kernel():
    rng = np.random.default_rng(0)
    n_disc = 8
    n_mc = 5
    dt = 0.1
    k1 = 1.3
    x0 = 0.2
    dx = rng.normal(size=(n_disc, n_mc))

    c_x1, _ = exp_decay_kernel_matrices(k1, 2.0, n_disc, dt)
    expected = c_x1 @ dx
    expected += x0 * np.exp(-k1 * dt * np.arange(1, n_disc + 1))[:, np.newaxis]
    expected = np.vstack([np.full((1, n_mc), x0), expected])

    actual = _ou_factor_paths(dx, np.exp(-k1 * dt), x0)

    assert np.allclose(actual, expected)


def test_two_bergomi_integrals_jit_matches_vectorized_reduction():
    model = TwoFactorBergomiModel(
        params=NONFLAT_PARAMS | {"x1_0": 0.1, "x2_0": -0.2},
        xi0=lambda t: 0.04 + 0.01 * np.asarray(t),
    )
    n_disc = 6
    n_mc = 5
    tab_t = np.linspace(0.0, 1.2, n_disc + 1)
    dt = tab_t[1] - tab_t[0]
    rng = np.random.default_rng(123)
    normal = rng.normal(size=(3, n_disc, n_mc))

    std_dx1 = (dt * _func_I(2.0 * model.k1 * dt)) ** 0.5
    std_dx2 = (dt * _func_I(2.0 * model.k2 * dt)) ** 0.5
    std_dW = dt**0.5
    cov_dx1_dx2 = model.rho12 * dt * _func_I((model.k1 + model.k2) * dt)
    cov_dx1_dW = model.rhoS1 * dt * _func_I(model.k1 * dt)
    cov_dx2_dW = model.rhoS2 * dt * _func_I(model.k2 * dt)

    l11 = std_dx1
    l21 = cov_dx1_dx2 / l11
    l31 = cov_dx1_dW / l11
    l22 = np.sqrt(std_dx2**2 - l21**2)
    l32 = (cov_dx2_dW - l21 * l31) / l22
    l33 = np.sqrt(std_dW**2 - l31**2 - l32**2)

    decay1 = np.exp(-model.k1 * dt)
    decay2 = np.exp(-model.k2 * dt)
    xi0_t = np.asarray(model.xi0(tab_t), dtype=float)
    variance_drift_t = np.asarray(
        0.5 * model.w**2 * model._var(tab_t, tab_t), dtype=float
    )
    weight1 = 1.0 - model.theta
    weight2 = model.theta
    x1_0_shifted = model.x1_0 + 1e-3 * model.rhoS1 / model.xi0_0**0.5
    x2_0_shifted = model.x2_0 + 1e-3 * model.rhoS2 / model.xi0_0**0.5
    n_disc_range = np.arange(n_disc + 1, dtype=float)
    shift_factor = np.exp(
        model.w
        * model.alpha
        * (
            weight1
            * (x1_0_shifted - model.x1_0)
            * np.exp(-model.k1 * dt * n_disc_range)
            + weight2
            * (x2_0_shifted - model.x2_0)
            * np.exp(-model.k2 * dt * n_disc_range)
        )
    )

    actual = _two_bergomi_integrals_from_normal_jit(
        normal,
        xi0_t,
        variance_drift_t,
        shift_factor,
        dt,
        l11,
        l21,
        l22,
        l31,
        l32,
        l33,
        decay1,
        decay2,
        model.x1_0,
        model.x2_0,
        model.w,
        model.alpha,
        weight1,
        weight2,
        True,
    )

    chol = np.array(
        [
            [l11, 0.0, 0.0],
            [l21, l22, 0.0],
            [l31, l32, l33],
        ]
    )
    dx1_dx2_dw = (chol @ normal.reshape(3, -1)).reshape(3, n_disc, n_mc)
    dx1 = dx1_dx2_dw[0]
    dx2 = dx1_dx2_dw[1]
    dw = dx1_dx2_dw[2]
    x1 = _ou_factor_paths(dx1, decay1, model.x1_0)
    x2 = _ou_factor_paths(dx2, decay2, model.x2_0)
    v_base = xi0_t[:, np.newaxis] * np.exp(
        model.w * model.alpha * (weight1 * x1 + weight2 * x2)
        - variance_drift_t[:, np.newaxis]
    )
    v_prev = v_base[:-1, :]
    v_next = v_base[1:, :]
    expected_int_sqrt_v_dw = np.sum(np.sqrt(v_prev) * dw, axis=0)
    expected_int_v_dt = 0.5 * dt * np.sum(v_prev + v_next, axis=0)
    v_ssr_prev = v_prev * shift_factor[:-1, np.newaxis]
    expected_int_sqrt_v_dw_shifted = np.sum(np.sqrt(v_ssr_prev) * dw, axis=0)
    expected_int_v_dt_shifted = (
        0.5 * dt * np.sum(v_ssr_prev + v_next * shift_factor[1:, np.newaxis], axis=0)
    )

    assert np.allclose(actual[0], expected_int_v_dt)
    assert np.allclose(actual[1], expected_int_sqrt_v_dw)
    assert np.allclose(actual[2], expected_int_v_dt_shifted)
    assert np.allclose(actual[3], expected_int_sqrt_v_dw_shifted)
