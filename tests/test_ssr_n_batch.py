import numpy as np
import pytest
from scipy import stats

from ssr import model as model_module
from ssr.model import _atm_impvol_from_path_integrals, _black_atm_impvol_from_price
from ssr.rough_bergomi import RoughBergomiModel
from ssr.utils import implied_vol_from_paths


def _flat_xi0(t):
    return 0.04 * np.ones_like(t)


def _make_model():
    return RoughBergomiModel(
        params={"eta": 1.0, "H": 0.3, "rho": -0.5},
        xi0=_flat_xi0,
    )


def test_ssr_mc_all_batches_mean_and_confidence_intervals(monkeypatch):
    model = _make_model()
    seed = 123
    n_batch = 3
    batch_seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, size=n_batch)
    seen_seeds = []

    def fake_simulate_mc(**kwargs):
        seed_value = int(kwargs["seed"])
        seen_seeds.append(seed_value)
        level = (seed_value % 1000) / 100.0
        return {
            "int_v_dt": np.full(4, level),
            "int_sqrt_v_dw": np.zeros(4),
            "int_v_dt_shifted": np.full(4, level + 0.2),
            "int_sqrt_v_dw_shifted": np.zeros(4),
        }

    def fake_atm_impvol_from_path_integrals(
        T,
        int_v_dt,
        int_sqrt_v_dw,
        s0,
        conditioning,
        rho_cond,
        return_skew,
    ):
        level = np.asarray(int_v_dt)[0]
        if return_skew:
            return level, level + 1.0
        return level

    monkeypatch.setattr(model, "simulate_mc", fake_simulate_mc)
    monkeypatch.setattr(
        model_module,
        "_atm_impvol_from_path_integrals",
        fake_atm_impvol_from_path_integrals,
    )
    beta_1_calls = 0

    def fake_beta_1_gauss_quad(T, n_quad):
        nonlocal beta_1_calls
        beta_1_calls += 1
        return np.array([2.0, 4.0])

    monkeypatch.setattr(model, "_beta_1_gauss_quad", fake_beta_1_gauss_quad)

    out = model.ssr_mc_all(
        T=np.array([0.5, 1.0]),
        n_mc=4,
        n_disc=2,
        n_loop=2,
        n_quad=8,
        eps_ssr=0.1,
        seed=seed,
        n_batch=n_batch,
    )

    assert beta_1_calls == 1
    assert seen_seeds == [int(s) for s in np.repeat(batch_seeds, 2)]

    levels = np.asarray([(int(s) % 1000) / 100.0 for s in batch_seeds])
    atm_skew_batch = levels[:, None] + 1.0
    ssr_fd_batch = 2.0 / atm_skew_batch
    ssr_fd_batch = np.repeat(ssr_fd_batch, 2, axis=1)
    ssr_vs_batch = np.array([2.0, 4.0]) / atm_skew_batch
    beta_batch = ssr_fd_batch * atm_skew_batch

    expected = {
        "ssr_fd": ssr_fd_batch,
        "ssr_vs": ssr_vs_batch,
        "beta": beta_batch,
        "atm_skew": np.repeat(atm_skew_batch, 2, axis=1),
    }
    for key, values in expected.items():
        mean = values.mean(axis=0)
        err = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(n_batch)
        assert np.allclose(out[key], mean)
        assert np.allclose(out[f"{key}_low"], mean - err)
        assert np.allclose(out[f"{key}_high"], mean + err)
        assert f"{key}_stderr" not in out


def test_ssr_mc_all_single_batch_returns_point_estimates_only(monkeypatch):
    model = _make_model()

    def fake_simulate_mc(**kwargs):
        return {
            "int_v_dt": np.full(4, 2.0),
            "int_sqrt_v_dw": np.zeros(4),
            "int_v_dt_shifted": np.full(4, 2.2),
            "int_sqrt_v_dw_shifted": np.zeros(4),
        }

    def fake_atm_impvol_from_path_integrals(
        T,
        int_v_dt,
        int_sqrt_v_dw,
        s0,
        conditioning,
        rho_cond,
        return_skew,
    ):
        level = np.asarray(int_v_dt)[0]
        if return_skew:
            return level, np.array(3.0)
        return level

    monkeypatch.setattr(model, "simulate_mc", fake_simulate_mc)
    monkeypatch.setattr(
        model_module,
        "_atm_impvol_from_path_integrals",
        fake_atm_impvol_from_path_integrals,
    )
    monkeypatch.setattr(model, "_beta_1_gauss_quad", lambda T, n_quad: np.array([6.0]))

    out = model.ssr_mc_all(T=1.0, n_mc=4, n_disc=2, eps_ssr=0.1)

    assert np.allclose(out["ssr_fd"], [2.0 / 3.0])
    assert np.allclose(out["ssr_vs"], [2.0])
    assert np.allclose(out["beta"], [2.0])
    assert np.allclose(out["atm_skew"], [3.0])
    assert not any(key.endswith(("_low", "_high", "_stderr")) for key in out)


@pytest.mark.parametrize("n_batch", [0, -1])
def test_ssr_mc_all_rejects_invalid_n_batch(n_batch):
    model = _make_model()

    with pytest.raises(ValueError, match="n_batch must be a positive integer"):
        model.ssr_mc_all(
            T=1.0,
            n_mc=4,
            n_disc=2,
            eps_ssr=0.1,
            n_batch=n_batch,
        )


def test_ssr_fukasawa_all_batches_mean_and_confidence_intervals(monkeypatch):
    model = _make_model()
    seed = 123
    n_batch = 3
    batch_seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, size=n_batch)

    def fake_simulate_mc(**kwargs):
        seed_value = int(kwargs["seed"])
        level = (seed_value % 1000) / 100.0
        # S_T = 1.0 for two paths, 1.0 + level for the other two, so that F, the
        # digit and the kernel term all depend deterministically on `level`.
        return {
            "int_v_dt": np.log1p(level) * np.array([0.0, 0.0, 0.0, 0.0]) + 0.01,
            "int_sqrt_v_dw": np.array([-level, -level, level, level]) * 0.1,
            "int_sqrt_v_k_dw": np.array([0.02, 0.02, 0.02, 0.02]),
            "int_v_k_dt": np.array([0.0, 0.0, 0.0, 0.0]),
        }

    monkeypatch.setattr(model, "simulate_mc", fake_simulate_mc)

    out = model.ssr_fukasawa_all(
        T=np.array([0.5, 1.0]),
        n_mc=4,
        n_disc=2,
        n_loop=1,
        seed=seed,
        n_batch=n_batch,
    )

    def _expected_ssr(batch_seed):
        level = (int(batch_seed) % 1000) / 100.0
        int_v_dt = np.array([0.01, 0.01, 0.01, 0.01])
        int_sqrt_v_dw = np.array([-level, -level, level, level]) * 0.1
        S_T = model.s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
        F = S_T.mean()
        indicator = S_T < F
        kernel = np.array([0.02, 0.02, 0.02, 0.02])
        X = -np.mean(indicator * S_T * kernel) / (2.0 * F * model.xi0_0**0.5)
        digit = np.mean(S_T >= F)
        atm_impvol = _black_atm_impvol_from_price(
            F=F, T=1.0, price=np.maximum(S_T - F, 0.0).mean()
        )
        d2 = -0.5 * atm_impvol
        Y = stats.norm.cdf(d2) - digit
        return X / Y

    expected_per_batch = np.array(
        [[_expected_ssr(bs), _expected_ssr(bs)] for bs in batch_seeds]
    )
    expected_mean = expected_per_batch.mean(axis=0)
    expected_err = 1.96 * expected_per_batch.std(axis=0, ddof=1) / np.sqrt(n_batch)

    assert np.allclose(out["ssr_fukasawa"], expected_mean)
    assert np.allclose(out["ssr_fukasawa_low"], expected_mean - expected_err)
    assert np.allclose(out["ssr_fukasawa_high"], expected_mean + expected_err)


def test_ssr_fukasawa_all_single_batch_returns_point_estimate_only(monkeypatch):
    model = _make_model()

    def fake_simulate_mc(**kwargs):
        return {
            "int_v_dt": np.array([0.01, 0.01, 0.01, 0.01]),
            "int_sqrt_v_dw": np.array([-0.1, -0.1, 0.1, 0.1]),
            "int_sqrt_v_k_dw": np.array([0.02, 0.02, 0.02, 0.02]),
            "int_v_k_dt": np.array([0.0, 0.0, 0.0, 0.0]),
        }

    monkeypatch.setattr(model, "simulate_mc", fake_simulate_mc)

    out = model.ssr_fukasawa_all(T=1.0, n_mc=4, n_disc=2)

    assert "ssr_fukasawa" in out
    assert not any(key.endswith(("_low", "_high", "_stderr")) for key in out)


@pytest.mark.parametrize("n_batch", [0, -1])
def test_ssr_fukasawa_all_rejects_invalid_n_batch(n_batch):
    model = _make_model()

    with pytest.raises(ValueError, match="n_batch must be a positive integer"):
        model.ssr_fukasawa_all(T=1.0, n_mc=4, n_disc=2, n_batch=n_batch)


@pytest.mark.parametrize("conditioning", [False, True])
def test_atm_impvol_from_path_integrals_matches_general_helper(conditioning):
    T = 1.0
    int_v_dt = np.array([0.01, 0.02, 0.035, 0.05, 0.08])
    int_sqrt_v_dw = np.array([-0.2, -0.07, 0.03, 0.11, 0.24])
    rho_cond = -0.5 if conditioning else None

    expected = implied_vol_from_paths(
        k=0.0,
        T=T,
        int_v_dt=int_v_dt,
        int_sqrt_v_dw=int_sqrt_v_dw,
        s0=1.0,
        conditioning=conditioning,
        return_skew=True,
        rho_cond=rho_cond,
    )
    actual = _atm_impvol_from_path_integrals(
        T=T,
        int_v_dt=int_v_dt,
        int_sqrt_v_dw=int_sqrt_v_dw,
        s0=1.0,
        conditioning=conditioning,
        rho_cond=rho_cond,
        return_skew=True,
    )

    assert np.allclose(np.ravel(actual[0]), np.ravel(expected[0]))
    assert np.allclose(np.ravel(actual[1]), np.ravel(expected[1]))
