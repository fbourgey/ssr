import numpy as np
from numba import float64, njit, vectorize
from scipy.integrate import quad_vec

from model import (
    ForwardVarianceModel,
    require_params,
    validate_interval,
    validate_nonnegative,
    validate_positive,
    validate_positive_n_quad,
)


def _validate_positive_maturity(T):
    if np.any(np.asarray(T) <= 0.0):
        raise ValueError("T must be positive.")


@njit(cache=True)
def _coef_charfunc_jit(u, T, kappa, rho, eta):
    al = -u * u / 2 - 1j * u / 2
    bet = kappa - rho * eta * 1j * u
    gam = eta**2 / 2
    d = np.sqrt(bet * bet - 4 * al * gam)
    rp = (bet + d) / (2 * gam)
    rm = (bet - d) / (2 * gam)
    g = rm / rp
    exp_minus_d_t = np.exp(-d * T)
    D = rm * (1 - exp_minus_d_t) / (1 - g * exp_minus_d_t)
    C = kappa * (rm * T - 2 / eta**2 * np.log((1 - g * exp_minus_d_t) / (1 - g)))
    return C, D


class HestonModel(ForwardVarianceModel):
    """Heston model."""

    def __init__(self, params, s0=1.0) -> None:
        """
        Initialize Heston model.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters:
            - eta: volatility of volatility
            - kappa: mean reversion speed
            - rho: correlation between Brownian motions
            - vbar: long-term variance
            - v: initial variance
        s0 : float, optional
            Initial stock price, by default 1.0
        """
        self.kappa, self.rho, self.eta, self.vbar, self.v = require_params(
            params, ("kappa", "rho", "eta", "vbar", "v")
        )

        super().__init__(
            params=params,
            xi0=lambda t: _xi0_heston(t=t, kappa=self.kappa, vbar=self.vbar, v=self.v),
            s0=s0,
        )
        self._check_params()

    def _check_params(self):
        """Validate Heston parameters."""
        validate_positive("eta", self.eta)
        validate_nonnegative("kappa", self.kappa)
        validate_interval("rho", self.rho, -1, 1)
        validate_nonnegative("vbar", self.vbar)
        validate_nonnegative("v", self.v)

    def _clone_with_params(self, **updates):
        """Clone the model with updated parameters."""
        params = self.params.copy()
        params.update(updates)
        return self.__class__(s0=self.s0, params=params)

    def lbd_xi(self, u, t):
        return self.eta * np.exp(-self.kappa * (u - t)) * self.xi0(t) ** 0.5

    def charfunc(self, u, T):
        """
        Compute the characteristic function of the Heston model E[exp(i * u * X_T)]
        where X_T = log(S_T/S_0) at u for time T.

        Parameters
        ----------
        u : float or array_like
            Points at which to evaluate the characteristic function
        T : float
            Maturity

        Returns
        -------
        complex or array_like
            Value of the characteristic function
        """
        C, D = self._coef_charfunc(u, T)
        return np.exp(C * self.vbar + D * self.v)

    def _coef_charfunc(self, u, T):
        return _coef_charfunc_jit(u, T, self.kappa, self.rho, self.eta)

    def impvol(self, k, T):
        return self.impvol_cf(k, T)

    def atm_skew(self, T):
        return self.atm_skew_cf(T)

    def sig_varswap(self, T, n_quad: int = 0):
        """
        Compute the fair strike of a total variance swap.

        It is given by E[int_0^T v_t dt] where v_t is the variance process.

        Parameters
        ----------
        T : array_like
            Time to maturity

        Returns
        -------
        array_like
            Fair strike of the variance swap
        """
        if n_quad < 0:
            raise ValueError("n_quad must be >= 0.")

        T = np.atleast_1d(T)
        return np.sqrt(self.vbar + (self.v - self.vbar) * _func_I(self.kappa * T))

    def _coef_charfunc_D(self, u, T):
        """
        D coefficient in the Heston characteristic function.

        Parameters
        ----------
        u : complex or array_like
            Fourier variable (may be complex).
        T : float
            Time to maturity.

        Returns
        -------
        complex or ndarray
            Value of D(u, tau).
        """
        _, D = self._coef_charfunc(u, T)
        return D

    def _coef_charfunc_C(self, u, T):
        """
        C coefficient in the Heston characteristic function.

        Parameters
        ----------
        u : complex or array_like
            Fourier variable (may be complex).
        T : float
            Time to maturity.

        Returns
        -------
        complex or ndarray
            Value of C(u, tau).
        """
        C, _ = self._coef_charfunc(u, T)
        return C

    def ssr_vs(self, T):
        """
        Compute the VS Skew-Stickiness Ratio (SSR) for the Heston model.

        Parameters
        ----------
        T : float or array_like
            Time to maturity

        Returns
        -------
        float or array_like
            VS SSR value(s)
        """
        T = np.atleast_1d(T)
        out = self.beta_1(T) / self.atm_skew_cf(T=T)
        return out

    def ssr_fd(self, T, eps: float):
        """
        Compute Skew-Stickiness Ratio (SSR) via finite difference.
        """
        v_shift = self.v + eps * self.rho * self.eta
        heston_shift = self._clone_with_params(v=v_shift)
        T = np.atleast_1d(T)

        def _ssr_fd_single(Ti):
            out = (heston_shift.impvol(k=0.0, T=Ti) - self.impvol(k=0.0, T=Ti)) / eps
            out /= self.atm_skew_cf(T=Ti)
            return out[0]

        return np.array([_ssr_fd_single(Ti) for Ti in T])

    def ssr_cf(self, T, **kwargs):
        """
        Compute Skew-Stickiness Ratio (SSR) via characteristic function.

        Parameters
        ----------
        T : float
            Time to maturity.

        Returns
        -------
        float
            SSR value (ratio of two integrals).
        """

        T = np.atleast_1d(T)

        def _ssr_cf_single(Ti):

            def _integrand(a):
                ca, da = self._coef_charfunc(a - 0.5j, Ti)
                cf = np.exp(da * self.v + ca * self.vbar)
                denom = a**2 + 0.25
                return np.array(
                    [
                        np.real(self.rho * self.eta * da * cf / denom),
                        np.imag(cf * a / denom),
                    ]
                )

            integral = quad_vec(_integrand, 0, np.inf)[0]
            return integral[0] / integral[1]

        return np.array([_ssr_cf_single(Ti) for Ti in T])

    def ssr_forest(self, T):
        """
        Forest expansion approximation of SSR for Heston when kappa=0 and vbar=v0.
        See Section 5.3.1 in [Friz, Gatheral, "Computing the SSR", QF, 2025]

        Parameters
        ----------
        T : float
            Time to maturity (must be > 0).

        Returns
        -------
        float
            Approximate SSR using the forest expansion up to O(tau).
        """

        T = np.atleast_1d(T)

        if self.vbar != self.v:
            raise ValueError("vbar must be equal to v0 for this approximation.")

        _validate_positive_maturity(T)

        if self.kappa != 0:
            raise ValueError("kappa must be equal to 0 for this approximation.")

        # Numerator expansion up to O(tau)
        num = (
            1
            + (1 / 8) * self.eta * self.rho * T
            + (1 / 24) * (self.eta**2) * T / self.v
            - (1 / 96) * (self.rho**2) * (self.eta**2) * T / self.v
        )
        # Denominator expansion up to O(tau)
        denom = (
            1
            - (1 / 24) * self.rho * self.eta * T
            + (1 / 8) * (self.eta**2) * T / self.v
            - (3 / 32) * (self.rho**2) * (self.eta**2) * T / self.v
        )

        return 2.0 * num / denom

    def ssr_all(self, T, **kwargs):
        """Compute SSR, VS SSR, beta, and ATM skew."""
        # precompute common quantities
        atm_skew = self.atm_skew(T)
        ssr_vs = self.beta_1(T) / atm_skew
        ssr_fd = self.ssr_fd(T, **kwargs)
        ssr_cf = self.ssr_cf(T, **kwargs)
        beta = ssr_fd * atm_skew

        return {
            "ssr_fd": ssr_fd,
            "ssr_cf": ssr_cf,
            "ssr_vs": ssr_vs,
            "beta": beta,
            "atm_skew": atm_skew,
        }

    def ssr_bg_all(self, T, n_quad: int = 0):
        """
        Compute SSR, VS SSR, beta, and ATM skew from the Bergomi-Guyon expansion
        up to second order.
        """
        T = np.atleast_1d(T)
        _validate_positive_maturity(T)

        # precompute common quantities
        _sig_vs = self.sig_varswap(T)
        _c_x_xi_bg = self.c_x_xi_bg(T, n_quad)
        beta_1 = self.beta_1(T)
        beta_2 = self.beta_2(T)

        # atm skew
        atm_skew_1 = _c_x_xi_bg / (2.0 * T**2 * _sig_vs**3)
        atm_skew_2 = (
            4 * self.c_mu_bg(T, n_quad) * T * _sig_vs**2 - 3 * _c_x_xi_bg**2
        ) / (8 * T**3 * _sig_vs**5)

        # ssr and vs ssr
        ssr_0 = beta_1 / atm_skew_1
        ssr_vs_1 = -beta_1 * atm_skew_2 / atm_skew_1**2
        ssr_1 = ssr_vs_1 + beta_2 / atm_skew_1

        return {
            "ssr_0": ssr_0,
            "ssr_1": ssr_1,
            "ssr_vs_1": ssr_vs_1,
            "beta_1": beta_1,
            "beta_2": beta_2,
            "atm_skew_1": atm_skew_1,
            "atm_skew_2": atm_skew_2,
        }

    def mu_bg(self, t, u) -> float | np.ndarray:
        return self.xi0(t) * self.rho * self.eta * np.exp(-self.kappa * (u - t))

    def _c_x_xi_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)
        return self._c_x_xi_bg_gauss_leg_quad(T, n_quad)

    def c_x_xi_bg(self, T, n_quad: int = 0) -> float | np.ndarray:
        if n_quad < 0:
            raise ValueError("n_quad must be >= 0.")

        if n_quad > 0:
            return self._c_x_xi_bg_gauss_quad(T, n_quad)

        T = np.atleast_1d(T)
        if self.kappa == 0.0:
            return self.rho * self.eta * self.v * T**2 / 2.0

        # closed-form expression
        I_kT = _func_I(self.kappa * T)
        out = self.vbar * (1 - I_kT) + (self.v - self.vbar) * (
            I_kT - np.exp(-self.kappa * T)
        )
        out *= self.rho * self.eta * T / self.kappa
        return out

    def c_mu_bg(self, T, n_quad: int = 0) -> float | np.ndarray:
        if n_quad < 0:
            raise ValueError("n_quad must be >= 0.")

        if n_quad > 0:
            return self._c_mu_bg_gauss_quad(T, n_quad)

        T = np.atleast_1d(T)
        if self.kappa == 0.0:
            return (self.rho * self.eta) ** 2 * self.v * T**3 / 6.0

        # closed-form expression
        A, B = _c_mu_bg_factors(self.kappa * T)
        out = A * self.v + B * (self.vbar - self.v)
        out *= (self.rho * self.eta) ** 2 * T**3 / 2.0
        return out

    def beta_1(self, T: float | np.ndarray) -> float | np.ndarray:
        T = np.atleast_1d(T)
        return (
            self.eta * self.rho * _func_I(self.kappa * T) / (2.0 * self.sig_varswap(T))
        )

    def _beta_1_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)
        return self.beta_1(T)

    def beta_2(self, T: float | np.ndarray) -> float | np.ndarray:
        T = np.atleast_1d(T)
        _validate_positive_maturity(T)

        _sig_vs = self.sig_varswap(T)
        out = self._dxi_c_x_xi_beta_2(T) * _sig_vs - self.c_x_xi_bg(T) * self.beta_1(T)
        out /= 4.0 * T * _sig_vs**2
        return out

    def _beta_2_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)
        return self.beta_2(T)

    def _dxi_c_x_xi_bg_gauss_quad(
        self,
        T: float | np.ndarray,
        t: float | np.ndarray,
        u: float | np.ndarray,
        n_quad: int,
    ) -> float | np.ndarray:
        if self.kappa == 0.0:
            return self.rho * self.eta * (T - u) / T

        out = self.rho * self.eta * (-np.expm1(-self.kappa * (T - u)))
        return out / (self.kappa * T)

    def _dxi_c_x_xi_beta_2_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)
        return self._dxi_c_x_xi_beta_2(T)

    def _dxi_c_x_xi_beta_2(self, T: float | np.ndarray) -> float | np.ndarray:
        T = np.atleast_1d(T)
        if self.kappa == 0.0:
            return (self.rho * self.eta) ** 2 * T**2 / 2.0

        out = _func_I(self.kappa * T) - np.exp(-self.kappa * T)
        out *= (self.rho * self.eta) ** 2 * T / self.kappa
        return out


def _xi0_heston(t, kappa, vbar, v):
    """
    Forward variance curve for the Heston model.

    Parameters
    ----------
    t : array_like
        Time to maturity
    kappa : float
        Mean reversion rate
    vbar : float
        Long-term variance
    v : float
        Initial variance
    Returns
    -------
    array_like
        Forward variance curve values at time t
    """
    validate_nonnegative("kappa", kappa)
    validate_nonnegative("vbar", vbar)
    validate_nonnegative("v", v)

    t = np.asarray(t, dtype=float)
    validate_nonnegative("t", np.min(t))

    if kappa == 0.0:
        return np.full_like(t, v)
    return vbar + (v - vbar) * np.exp(-kappa * t)


def get_params_heston(id: int):
    """Get Heston parameters for a given id."""
    if id == 1:
        # Bergomi (2016), Stochastic Volatility Modeling, Chapter 6, page 211
        # Initial variance v in [0.01, 0.04, 0.16]
        return {"kappa": 1.0, "rho": -0.8, "eta": 0.6, "vbar": 0.04, "v": 0.04}
    else:
        raise ValueError("Invalid id. Please choose a valid id.")


@vectorize([float64(float64)], nopython=True, cache=True)
def _func_I(x):
    if x != 0.0:
        return -np.expm1(-x) / x
    return 1.0


def _c_mu_bg_factors(x):
    x = np.asarray(x, dtype=float)
    out_a, out_b = _c_mu_bg_factors_jit(np.ravel(x))
    return out_a.reshape(x.shape), out_b.reshape(x.shape)


@njit(cache=True)
def _c_mu_bg_factors_jit(x):
    out_a = np.empty_like(x)
    out_b = np.empty_like(x)
    for i in range(x.size):
        xi = x[i]
        if abs(xi) < 1e-4:
            out_a[i] = (
                1 / 3 - xi / 6 + xi**2 / 20 - xi**3 / 90 + xi**4 / 504 - xi**5 / 3360
            )
            out_b[i] = xi / 12 - xi**2 / 20 + xi**3 / 60 - xi**4 / 252 + xi**5 / 1344
        else:
            exp_minus_x = np.exp(-xi)
            out_a[i] = 2 * (xi - 2 + (2 + xi) * exp_minus_x) / xi**3
            out_b[i] = (2 * xi - 6 + (6 + 4 * xi + xi**2) * exp_minus_x) / xi**3
    return out_a, out_b
