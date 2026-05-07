import numpy as np
from scipy.interpolate import interp1d

__all__ = (
    "available_dates",
    "available_models",
    "get_params",
    "get_xi0_interpolator",
)

DATES = ("2024-05-06", "2024-08-05", "2025-04-07")  # calibration dates

_MODEL_PARAMS = {
    DATES[0]: {
        "heston": {
            "v": 0.011,
            "kappa": 1.1,
            "vbar": 0.043,
            "eta": 0.24,
            "rho": -1.0,
        },
        "rough_heston": {
            "H": 0.38,
            "kappa": 3.44e-4,
            "eta": 0.16,
            "rho": -0.91,
        },
        "two_bergomi": {
            "w": 5.0,
            "k1": 100.0,
            "k2": 0.30,
            "theta": 0.36,
            "rho12": 0.222,
            "rhoS1": -0.613,
            "rhoS2": -0.498,
        },
        "rough_bergomi": {
            "H": 0.23,
            "eta": 1.50,
            "rho": -0.62,
        },
    },
    DATES[1]: {
        "heston": {
            "v": 0.117,
            "kappa": 3.37,
            "vbar": 0.048,
            "eta": 1.99,
            "rho": -0.68,
        },
        "rough_heston": {
            "H": 0.17,
            "kappa": 1.69,
            "eta": 0.53,
            "rho": -0.99,
        },
        "two_bergomi": {
            "w": 8.99,
            "k1": 14.56,
            "k2": 2.12,
            "theta": 0.32,
            "rho12": -0.69,
            "rhoS1": -0.47,
            "rhoS2": -0.31,
        },
        "rough_bergomi": {
            "H": 0.11,
            "eta": 2.39,
            "rho": -0.86,
        },
    },
    DATES[2]: {
        "heston": {
            "v": 0.39,
            "kappa": 1.38,
            "vbar": 0.20,
            "eta": 7.71,
            "rho": -0.48,
        },
        "rough_heston": {
            "H": 0.024,
            "kappa": 0.43,
            "eta": 0.99,
            "rho": -0.62,
        },
        "two_bergomi": {
            "w": 11.5,
            "k1": 24.5,
            "k2": 2.1,
            "theta": 0.31,
            "rho12": -0.70,
            "rhoS1": -0.47,
            "rhoS2": -0.30,
        },
        "rough_bergomi": {
            "H": 0.097,
            "eta": 2.08,
            "rho": -0.99,
        },
    },
}

_MATS = (7 / 365, 1 / 12, 3 / 12, 6 / 12, 9 / 12, 1, 1.5, 2, 2.5, 3)

_XI0 = {
    DATES[0]: {
        "rough_heston": (
            0.0097,
            0.0153,
            0.0178,
            0.0228,
            0.0270,
            0.0299,
            0.0342,
            0.0389,
            0.0423,
            0.0469,
        ),
        "two_bergomi": (
            0.0099,
            0.0161,
            0.0192,
            0.0300,
            0.0440,
            0.0540,
            0.0790,
            0.1100,
            0.1400,
            0.1800,
        ),
        "rough_bergomi": (
            0.0101,
            0.0164,
            0.0197,
            0.0277,
            0.0319,
            0.0411,
            0.0450,
            0.0587,
            0.0679,
            0.0830,
        ),
    },
    DATES[1]: {
        "rough_heston": (
            0.1202,
            0.0861,
            0.0651,
            0.0511,
            0.0414,
            0.0380,
            0.0379,
            0.0324,
            0.0378,
            0.0368,
        ),
        "two_bergomi": (
            0.1268,
            0.1128,
            0.0834,
            0.0708,
            0.0808,
            0.0771,
            0.0671,
            0.0243,
            0.0604,
            0.0604,
        ),
        "rough_bergomi": (
            0.1396,
            0.1003,
            0.0801,
            0.0665,
            0.0551,
            0.0605,
            0.0623,
            0.0566,
            0.0797,
            0.0800,
        ),
    },
    DATES[2]: {
        "rough_heston": (
            0.4258,
            0.1909,
            0.1310,
            0.1012,
            0.0781,
            0.0761,
            0.0685,
            0.0614,
            0.0635,
            0.0704,
        ),
        "two_bergomi": (
            0.4000,
            0.1800,
            0.1400,
            0.1200,
            0.1000,
            0.0800,
            0.0800,
            0.0800,
            0.0800,
            0.0800,
        ),
        "rough_bergomi": (
            0.4000,
            0.1154,
            0.0959,
            0.0608,
            0.1018,
            0.0618,
            0.0399,
            0.0692,
            0.1045,
            0.1014,
        ),
    },
}


def _format_options(options):
    return ", ".join(sorted(options))


def _lookup(table, date, model, table_name):
    if date not in table:
        raise ValueError(
            f"Unknown calibration date {date!r} for {table_name}. "
            f"Available dates: {_format_options(table)}."
        )
    if model not in table[date]:
        raise ValueError(
            f"Unknown model {model!r} for {table_name} on {date!r}. "
            f"Available models: {_format_options(table[date])}."
        )
    return table[date][model]


def _validate_data():
    mats = np.asarray(_MATS, dtype=float)
    if mats.ndim != 1 or len(mats) == 0:
        raise ValueError("_MATS must be a non-empty one-dimensional grid.")
    if not np.all(np.diff(mats) > 0.0):
        raise ValueError("_MATS must be strictly increasing.")

    missing_param_dates = set(_XI0) - set(_MODEL_PARAMS)
    if missing_param_dates:
        raise ValueError(
            "xi0 curves reference dates without model parameters: "
            f"{_format_options(missing_param_dates)}."
        )

    for date, curves_by_model in _XI0.items():
        missing_models = set(curves_by_model) - set(_MODEL_PARAMS[date])
        if missing_models:
            raise ValueError(
                f"xi0 curves for {date!r} reference models without parameters: "
                f"{_format_options(missing_models)}."
            )

        for model, curve in curves_by_model.items():
            xi0 = np.asarray(curve, dtype=float)
            if xi0.shape != mats.shape:
                raise ValueError(
                    f"xi0 curve for {date!r}/{model!r} has shape {xi0.shape}; "
                    f"expected {mats.shape}."
                )
            if not np.all(np.isfinite(xi0)):
                raise ValueError(f"xi0 curve for {date!r}/{model!r} is not finite.")
            if not np.all(xi0 > 0.0):
                raise ValueError(f"xi0 curve for {date!r}/{model!r} must be positive.")


def available_dates():
    """Return the calibration dates with model parameters."""
    return tuple(sorted(_MODEL_PARAMS))


def available_models(date=None, *, xi0_only=False):
    """
    Return available models for a calibration date.

    Parameters
    ----------
    date : str, optional
        Calibration date. If omitted, return all models with parameters.
    xi0_only : bool, optional
        If True, return only models with an initial forward variance curve.
    """
    table = _XI0 if xi0_only else _MODEL_PARAMS
    if date is None:
        models = set()
        for models_by_date in table.values():
            models.update(models_by_date)
        return tuple(sorted(models))

    table_name = "xi0 curves" if xi0_only else "parameters"
    if date not in table:
        raise ValueError(
            f"Unknown calibration date {date!r} for {table_name}. "
            f"Available dates: {_format_options(table)}."
        )
    return tuple(sorted(table[date]))


def get_params(date, model):
    """
    Retrieve the calibration parameters for a given date and model.

    Parameters
    ----------
    date : str
        The date of the calibration in 'YYYY-MM-DD' format.
    model : str
        The name of the model (e.g., 'heston', 'rough_heston', 'two_bergomi',
        'rough_bergomi').

    Returns
    -------
    dict
        A dictionary containing the calibration parameters for the specified date and
        model.
    """
    return _lookup(_MODEL_PARAMS, date, model, "parameters").copy()


def get_xi0_interpolator(date, model, kind="next"):
    """
    Retrieve an interpolator for the initial forward variance curve for a given date
    and model.

    Parameters
    ----------
    date : str
        The date of the calibration in 'YYYY-MM-DD' format.
    model : str
        The name of the model (e.g., 'rough_heston', 'two_bergomi', 'rough_bergomi').
    kind : str, optional
        The type of interpolation to perform. Default is 'next', which corresponds to
        piecewise constant interpolation using the next provided data points. Other
        include "cubic", "linear", etc.

    Returns
    -------
    interp1d
        An interpolator for the initial forward variance curve.
    """
    mats = np.asarray(_MATS, dtype=float)
    xi0 = np.asarray(_lookup(_XI0, date, model, "xi0 curves"), dtype=float)
    return interp1d(
        mats,
        xi0,
        kind=kind,
        bounds_error=False,
        fill_value=(xi0[0], xi0[-1]),  # pyright: ignore[reportArgumentType]
    )


_validate_data()
