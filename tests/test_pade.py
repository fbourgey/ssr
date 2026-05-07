import numpy as np
import pytest

import pade


@pytest.mark.parametrize(
    ("n_pade", "direct_func"),
    [
        (2, pade._h_pade_22),
        (3, pade._h_pade_33),
        (4, pade._h_pade_44),
        (5, pade._h_pade_55),
        (6, pade._h_pade_66),
    ],
)
def test_h_pade_dispatch_matches_direct_implementation(n_pade, direct_func):
    params = {"eta": 0.2, "H": 0.05, "kappa": 0.0, "rho": -0.65}
    tau = np.array([0.05, 0.5, 2.0])
    a = 0.7 - 0.5j

    assert np.allclose(
        pade._h_pade(tau=tau, a=a, params=params, n_pade=n_pade),
        direct_func(tau=tau, a=a, params=params),
    )


def test_h_pade_rejects_invalid_order():
    params = {"eta": 0.4, "H": 0.1, "kappa": 1.0, "rho": -0.65}

    with pytest.raises(ValueError, match="Invalid Padé order"):
        pade._h_pade(tau=0.5, a=0.7 - 0.5j, params=params, n_pade=7)
