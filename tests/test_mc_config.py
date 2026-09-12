import numpy as np
import pytest

from ssr.model import MonteCarloConfig, SsrMonteCarloConfig
from ssr.rough_bergomi import RoughBergomiModel
from ssr.two_bergomi import TwoFactorBergomiModel


def _flat_xi0(t):
    return 0.04 * np.ones_like(t)


@pytest.fixture(
    params=[
        RoughBergomiModel(params={"eta": 1.0, "H": 0.2, "rho": -0.5}, xi0=_flat_xi0),
        TwoFactorBergomiModel(
            params={
                "k1": 1.0,
                "k2": 2.0,
                "theta": 0.5,
                "w": 1.0,
                "rho12": 0.0,
                "rhoS1": -0.5,
                "rhoS2": -0.3,
            },
            xi0=_flat_xi0,
        ),
    ]
)
def mc_model(request):
    return request.param


def test_impvol_accepts_config_object(mc_model, monkeypatch):
    captured = {}

    def fake_impvol_mc(**kwargs):
        captured.update(kwargs)
        return "impvol"

    monkeypatch.setattr(mc_model, "impvol_mc", fake_impvol_mc)

    config = MonteCarloConfig(n_mc=10, n_disc=4, seed=7, conditioning=True)
    assert mc_model.impvol(k=0.0, T=1.0, config=config, return_skew=True) == "impvol"

    assert captured["n_mc"] == 10
    assert captured["n_disc"] == 4
    assert captured["seed"] == 7
    assert captured["conditioning"] is True
    assert captured["return_skew"] is True
    assert captured["rho_cond"] is not None


def test_impvol_keeps_legacy_keyword_api(mc_model, monkeypatch):
    captured = {}

    def fake_impvol_mc(**kwargs):
        captured.update(kwargs)
        return "impvol"

    monkeypatch.setattr(mc_model, "impvol_mc", fake_impvol_mc)

    assert mc_model.impvol(k=0.0, T=1.0, n_mc=10, n_disc=4) == "impvol"

    assert captured["n_mc"] == 10
    assert captured["n_disc"] == 4
    assert captured["n_loop"] == 1
    assert captured["conditioning"] is False
    assert captured["rho_cond"] is None


def test_ssr_all_accepts_config_object(mc_model, monkeypatch):
    captured = {}

    def fake_ssr_mc_all(**kwargs):
        captured.update(kwargs)
        return "ssr"

    monkeypatch.setattr(mc_model, "ssr_mc_all", fake_ssr_mc_all)

    config = SsrMonteCarloConfig(
        n_mc=10,
        n_disc=4,
        n_quad=8,
        eps_ssr=1e-3,
        seed=7,
        conditioning=True,
        n_batch=3,
    )
    assert mc_model.ssr_all(T=1.0, config=config) == "ssr"

    assert captured["n_mc"] == 10
    assert captured["n_disc"] == 4
    assert captured["n_quad"] == 8
    assert captured["eps_ssr"] == 1e-3
    assert captured["seed"] == 7
    assert captured["conditioning"] is True
    assert captured["n_batch"] == 3
    assert captured["rho_cond"] is not None


def test_ssr_all_keeps_legacy_keyword_api(mc_model, monkeypatch):
    captured = {}

    def fake_ssr_mc_all(**kwargs):
        captured.update(kwargs)
        return "ssr"

    monkeypatch.setattr(mc_model, "ssr_mc_all", fake_ssr_mc_all)

    assert mc_model.ssr_all(T=1.0, n_mc=10, n_disc=4, n_quad=8, eps_ssr=1e-3) == "ssr"

    assert captured["n_mc"] == 10
    assert captured["n_disc"] == 4
    assert captured["n_quad"] == 8
    assert captured["eps_ssr"] == 1e-3
    assert captured["conditioning"] is False
    assert captured["n_batch"] == 1
    assert captured["rho_cond"] is None


def test_ssr_all_accepts_n_batch_keyword(mc_model, monkeypatch):
    captured = {}

    def fake_ssr_mc_all(**kwargs):
        captured.update(kwargs)
        return "ssr"

    monkeypatch.setattr(mc_model, "ssr_mc_all", fake_ssr_mc_all)

    assert (
        mc_model.ssr_all(
            T=1.0,
            n_mc=10,
            n_disc=4,
            n_quad=8,
            eps_ssr=1e-3,
            n_batch=4,
        )
        == "ssr"
    )

    assert captured["n_batch"] == 4


def test_ssr_all_rejects_removed_mc_error_keyword(mc_model):
    with pytest.raises(TypeError, match="Unexpected SSR Monte Carlo config values"):
        mc_model.ssr_all(
            T=1.0,
            n_mc=10,
            n_disc=4,
            n_quad=8,
            eps_ssr=1e-3,
            mc_error=True,
        )
