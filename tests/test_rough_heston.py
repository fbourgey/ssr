import pytest
import numpy as np
from ssr import rough_heston
from ssr.rough_heston import RoughHestonModel


@pytest.mark.parametrize("Ts", [np.linspace(1e-3, 5.0, 5)])
@pytest.mark.parametrize("eta", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("H", [0.1, 0.3, 0.5])
@pytest.mark.parametrize("rho", [-0.9, -0.5, -0.2])
@pytest.mark.parametrize("xi0", [lambda t: 0.04 * np.ones_like(t)])
def test_bg_beta_skew(Ts, eta, H, rho, xi0):
    """
    Check closed-form formulas for beta_1, beta_2, atm_skew_1, and atm_skew_2 when the
    initial forward variance curve is flat.
    """
    # NOTE: can be a bit slow because of Mittag-Leffler function evaluations, so we
    # may use fewer time points than in the other tests.
    params = {"eta": eta, "H": H, "rho": rho, "kappa": 0.0}
    rheston = RoughHestonModel(params=params, xi0=xi0)

    n_quad = 60
    out_bg = rheston.ssr_bg_all(Ts, n_quad=n_quad)

    assert np.allclose(out_bg["beta_1"], rheston._beta_1_bg_flat_kappa_zero(Ts))
    assert np.allclose(out_bg["beta_2"], rheston._beta_2_bg_flat_kappa_zero(Ts))
    assert np.allclose(out_bg["atm_skew_1"], rheston._atm_skew_1_bg_flat_kappa_zero(Ts))
    assert np.allclose(out_bg["atm_skew_2"], rheston._atm_skew_2_bg_flat_kappa_zero(Ts))


def test_charfunc_vectorized_maturities_match_scalar_calls():
    params = {"eta": 0.4, "H": 0.1, "rho": -0.65, "kappa": 1.0}
    rheston = RoughHestonModel(params=params, xi0=lambda t: 0.04 * np.ones_like(t))
    Ts = np.array([0.25, 1.0])

    vectorized = rheston.charfunc(u=0.7 - 0.5j, T=Ts, n_pade=2, n_quad=5)
    scalar = np.array(
        [rheston.charfunc(u=0.7 - 0.5j, T=T, n_pade=2, n_quad=5)[0] for T in Ts]
    )

    assert vectorized.shape == Ts.shape
    assert np.allclose(vectorized, scalar)


def test_charfunc_rejects_invalid_pade_order():
    params = {"eta": 0.4, "H": 0.1, "rho": -0.65, "kappa": 1.0}
    rheston = RoughHestonModel(params=params, xi0=lambda t: 0.04 * np.ones_like(t))

    with pytest.raises(ValueError, match="Invalid Padé order"):
        rheston.charfunc(u=0.7 - 0.5j, T=1.0, n_pade=7, n_quad=5)


def test_atm_skew_forwards_characteristic_function_kwargs(monkeypatch):
    params = {"eta": 0.4, "H": 0.1, "rho": -0.65, "kappa": 1.0}
    rheston = RoughHestonModel(params=params, xi0=lambda t: 0.04 * np.ones_like(t))
    captured = {}

    def fake_atm_skew_cf(T, **kwargs):
        captured["T"] = T
        captured["kwargs"] = kwargs
        return np.array([1.23])

    monkeypatch.setattr(rheston, "atm_skew_cf", fake_atm_skew_cf)

    assert np.allclose(rheston.atm_skew(1.0, n_pade=2, n_quad=5), np.array([1.23]))
    assert captured == {"T": 1.0, "kwargs": {"n_pade": 2, "n_quad": 5}}


def test_c_x_xi_gauss_matches_scipy_quad():
    params = {"eta": 0.8, "H": 0.3, "rho": -0.7, "kappa": 1.3}
    rheston = RoughHestonModel(
        params=params,
        xi0=lambda t: 0.04 * np.ones_like(t),
    )
    Ts = np.array([0.2, 0.5])

    assert np.allclose(
        rheston._c_x_xi_bg_gauss_quad(Ts, n_quad=25),
        rheston._c_x_xi_bg_scipy_quad(Ts),
        rtol=5e-6,
        atol=1e-9,
    )


@pytest.mark.parametrize(
    ("H", "ssr_rtol"),
    [
        (0.1, 1.2e-2),
        (0.2, 4.5e-3),
        (0.3, 1.5e-3),
        (0.4, 3e-3),
    ],
)
def test_characteristic_function_matches_finite_difference(H, ssr_rtol):
    params = {"eta": 0.8, "H": H, "rho": -0.7, "kappa": 1.3}
    rheston = RoughHestonModel(
        params=params,
        xi0=lambda t: 0.04 * np.ones_like(t),
    )
    Ts = np.array([0.1, 0.7, 2.0])
    kwargs = {"n_pade": 5, "n_quad": 40}

    ssr_cf = rheston.ssr_cf(Ts, **kwargs)
    atm_skew_cf = rheston.atm_skew_cf(Ts, **kwargs)

    for eps in (1e-4, 1e-5):
        assert np.allclose(
            ssr_cf,
            rheston.ssr_fd(Ts, eps=eps, **kwargs),
            rtol=ssr_rtol,
            atol=1e-6,
        )
        assert np.allclose(
            atm_skew_cf,
            rheston.atm_skew_fd(Ts, eps=eps, **kwargs),
            rtol=5e-5,
            atol=1e-8,
        )


def test_ml_two_caches_repeated_array_values(monkeypatch):
    rough_heston.clear_ml_two_cache()
    calls = 0

    def fake_mittag_leffler(z, alpha, beta):
        nonlocal calls
        calls += 1
        return np.asarray(z) + alpha + beta

    monkeypatch.setattr(rough_heston, "mittag_leffler", fake_mittag_leffler)
    z = np.array([[0.0, -0.5], [-1.0, -1.5]])

    first = rough_heston.ml_two(z, alpha=0.6)
    first[0, 0] = 999.0
    second = rough_heston.ml_two(z, alpha=0.6)

    assert calls == 1
    assert np.allclose(second, z + 1.2)
    assert second[0, 0] != first[0, 0]
    rough_heston.clear_ml_two_cache()


def test_ml_two_caches_repeated_scalar_values(monkeypatch):
    rough_heston.clear_ml_two_cache()
    calls = 0

    def fake_mittag_leffler(z, alpha, beta):
        nonlocal calls
        calls += 1
        return z + alpha + beta

    monkeypatch.setattr(rough_heston, "mittag_leffler", fake_mittag_leffler)

    first = rough_heston.ml_two(-0.5, alpha=0.6)
    second = rough_heston.ml_two(-0.5, alpha=0.6)

    assert calls == 1
    assert first == second == 0.7
    rough_heston.clear_ml_two_cache()
