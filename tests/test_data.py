import numpy as np
import pytest

from ssr.data import (
    available_dates,
    available_models,
    get_params,
    get_xi0_interpolator,
)


def test_available_dates_and_models_are_sorted():
    assert available_dates() == ("2024-05-06", "2024-08-05", "2025-04-07")
    assert available_models("2024-05-06") == (
        "heston",
        "rough_bergomi",
        "rough_heston",
        "two_bergomi",
    )
    assert available_models("2024-05-06", xi0_only=True) == (
        "rough_bergomi",
        "rough_heston",
        "two_bergomi",
    )


def test_get_params_returns_a_defensive_copy():
    params = get_params("2024-05-06", "heston")
    params["v"] = 0.0

    assert get_params("2024-05-06", "heston")["v"] == 0.011


def test_get_xi0_interpolator_uses_next_curve_point_and_flat_extrapolation():
    xi0 = get_xi0_interpolator("2024-05-06", "rough_heston")

    values = xi0(np.array([0.0, 0.02, 0.10, 4.0]))

    assert np.allclose(values, [0.0097, 0.0153, 0.0178, 0.0469])


def test_unknown_date_and_model_errors_are_actionable():
    with pytest.raises(ValueError, match="Available dates"):
        get_params("2020-01-01", "heston")

    with pytest.raises(ValueError, match="Available models"):
        get_xi0_interpolator("2024-05-06", "heston")
