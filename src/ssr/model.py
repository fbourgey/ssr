from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np
from scipy import stats
from scipy.integrate import quad

from .utils import (
    black_impvol,
    black_price,
    gauss_legendre,
    implied_vol_from_paths,
    lewis_formula_otm_price,
)


@dataclass(frozen=True)
class MonteCarloConfig:
    n_mc: int
    n_disc: int
    n_loop: int = 1
    seed: int | None = None
    conditioning: bool = False


@dataclass(frozen=True)
class SsrMonteCarloConfig(MonteCarloConfig):
    n_quad: int = 20
    eps_ssr: float = 1e-3
    n_batch: int = 1


def require_params(params: dict, required: tuple[str, ...]) -> tuple:
    """Return required parameter values or raise a consistent error."""
    missing = [p for p in required if p not in params]
    if missing:
        raise ValueError(f"Missing parameters: {missing}.")
    return tuple(params[p] for p in required)


def validate_positive(name: str, value: float) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def validate_positive_n_quad(n_quad: int) -> None:
    if n_quad <= 0:
        raise ValueError("n_quad must be > 0.")


def quad_complex(func: Callable, a: float, b: float, **kwargs) -> complex:
    """Integrate a possibly complex scalar function with scipy quad."""
    real = quad(lambda x: np.real(func(x)), a, b, **kwargs)[0]
    imag = quad(lambda x: np.imag(func(x)), a, b, **kwargs)[0]
    return real + 1j * imag


def validate_nonnegative(name: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0.")


def validate_interval(
    name: str,
    value: float,
    lower: float,
    upper: float,
    *,
    closed: bool = True,
) -> None:
    if closed:
        valid = lower <= value <= upper
        bounds = f"[{lower}, {upper}]"
    else:
        valid = lower < value < upper
        bounds = f"({lower}, {upper})"
    if not valid:
        raise ValueError(f"{name} must be in {bounds}.")


def resolve_mc_config(
    config: MonteCarloConfig | None,
    overrides: dict,
) -> MonteCarloConfig:
    if config is None:
        required = ("n_mc", "n_disc")
        missing = [p for p in required if p not in overrides]
        if missing:
            raise ValueError(f"Missing Monte Carlo config values: {missing}.")
        config = MonteCarloConfig(
            n_mc=overrides.pop("n_mc"),
            n_disc=overrides.pop("n_disc"),
        )
    if not isinstance(config, MonteCarloConfig):
        raise TypeError("config must be a MonteCarloConfig instance.")

    allowed = ("n_loop", "seed", "conditioning")
    updates = {key: overrides.pop(key) for key in list(overrides) if key in allowed}
    if overrides:
        raise TypeError(f"Unexpected Monte Carlo config values: {sorted(overrides)}.")
    if updates:
        config = replace(config, **updates)
    return config


def resolve_ssr_mc_config(
    config: SsrMonteCarloConfig | None,
    overrides: dict,
) -> SsrMonteCarloConfig:
    if config is None:
        required = ("n_mc", "n_disc", "n_quad", "eps_ssr")
        missing = [p for p in required if p not in overrides]
        if missing:
            raise ValueError(f"Missing SSR Monte Carlo config values: {missing}.")
        config = SsrMonteCarloConfig(
            n_mc=overrides.pop("n_mc"),
            n_disc=overrides.pop("n_disc"),
            n_quad=overrides.pop("n_quad"),
            eps_ssr=overrides.pop("eps_ssr"),
        )
    if not isinstance(config, SsrMonteCarloConfig):
        raise TypeError("config must be a SsrMonteCarloConfig instance.")

    allowed = ("n_loop", "seed", "conditioning", "n_quad", "eps_ssr", "n_batch")
    updates = {key: overrides.pop(key) for key in list(overrides) if key in allowed}
    if overrides:
        raise TypeError(
            f"Unexpected SSR Monte Carlo config values: {sorted(overrides)}."
        )
    if updates:
        config = replace(config, **updates)
    return config


def _black_atm_impvol_from_price(F: float, T: float, price: float) -> float:
    if T <= 0.0 or F <= 0.0 or not np.isfinite(price):
        return np.nan

    normalized_price = price / F
    if normalized_price < 0.0 or normalized_price >= 1.0:
        return np.nan

    return 2.0 * stats.norm.ppf(0.5 * (normalized_price + 1.0)) / T**0.5


def _atm_impvol_from_path_integrals(
    T: float,
    int_v_dt: np.ndarray,
    int_sqrt_v_dw: np.ndarray,
    s0: float,
    conditioning: bool,
    rho_cond: float | None,
    return_skew: bool,
):
    int_v_dt = np.asarray(int_v_dt, dtype=float).flatten()
    int_sqrt_v_dw = np.asarray(int_sqrt_v_dw, dtype=float).flatten()
    if int_v_dt.shape != int_sqrt_v_dw.shape:
        raise ValueError("int_v_dt and int_sqrt_v_dw must have the same shape.")

    if conditioning:
        if rho_cond is None:
            raise ValueError("rho_cond must be provided when conditioning is True.")
        if not (-1.0 <= rho_cond <= 1.0):
            raise ValueError("rho_cond must be between -1 and 1.")

        s0_cond = s0 * np.exp(-0.5 * rho_cond**2 * int_v_dt + rho_cond * int_sqrt_v_dw)
        vol_cond = np.sqrt((1.0 - rho_cond**2) * int_v_dt / T)
        F = s0_cond.mean()
        K = F
        price = black_price(K=K, T=T, F=s0_cond, vol=vol_cond, opttype=1.0).mean()
        impvol = _black_atm_impvol_from_price(F=F, T=T, price=price)

        if return_skew:
            w_cond = vol_cond * T**0.5
            with np.errstate(divide="ignore", invalid="ignore"):
                d2_cond = np.log(s0_cond / K) / w_cond - 0.5 * w_cond
            digit = stats.norm.cdf(d2_cond).mean()
    else:
        S = s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
        F = S.mean()
        price = np.maximum(S - F, 0.0).mean()
        impvol = _black_atm_impvol_from_price(F=F, T=T, price=price)

        if return_skew:
            digit = np.mean(S >= F)

    if return_skew:
        w = impvol * T**0.5
        d2 = -0.5 * w
        skew = (stats.norm.cdf(d2) - digit) / (stats.norm.pdf(d2) * T**0.5)
        return impvol, skew

    return impvol


class ForwardVarianceModel(ABC):
    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray] = lambda t: 0.2**2 * np.ones_like(t),
        s0: float = 1.0,
    ) -> None:
        """
        Initialize ForwardVarianceModel.

        Parameters
        ----------
        params : dict
            Model-specific parameters.
        xi0 : callable, optional
            Forward variance curve function xi0(t), must return positive values for
            t >= 0.
        s0 : float
            Initial spot price (must be positive).
        """
        if s0 <= 0.0:
            raise ValueError("Initial spot price s0 must be positive.")

        if not callable(xi0):
            raise ValueError("xi0 must be a callable function.")

        self.xi0 = xi0
        self._is_xi0_positive()
        self.xi0_0 = self.xi0(np.zeros(1))[0]
        self.xi0_flat = self._is_xi0_flat()
        self.params = params
        self.s0 = s0

    def __repr__(self) -> str:
        """Return a string representation of the model with its parameters."""
        msg = f"{self.__class__.__name__} with parameters:\n"
        for key in self.params:
            msg += f"{key}={self.params[key]}, "
        msg += f"s0={self.s0}"
        return msg

    @abstractmethod
    def lbd_xi(self, u, t) -> float | np.ndarray:
        """
        Compute the kernel function lbd_xi(u, t) where the forward variance
        process xi_t is given by

        dxi_t^u = lbd_xi(u, t, x_t^{v(t,u)}) ⋅ dW_t

        where W is a potentially multi-dimensional Brownian motion and v(u, t)
        is a function that determines the dependence of the kernel on the forward
        variance curve. In most examples, v(t,u) = t (Heston-type models) or
        v(t,u) = u (Bergomi-type models).

        Parameters
        ----------
        u : float or np.ndarray
            Upper time(s) (must satisfy u >= t).
        t : float or np.ndarray
            Lower time(s).

        Returns
        -------
        float or np.ndarray
            Value(s) of the kernel function evaluated at (u, t).
        """
        pass

    def _fukasawa_kernel_right_point(self, tab_t) -> np.ndarray:
        """
        Right-point kernel values kappa_i = k(t_{i+1}), for i = 0, ...,
        n_disc - 1, where k = sum_i rho_i k_i is the kernel of Theorem 2
        (Eq. 12) in Fukasawa (2026), "On the Skew Stickiness Ratio"
        (arXiv:2602.05241), for the Bergomi-type model of Eq. (11) in that paper.

        The right-point convention avoids k(0), which is singular for kernels such
        as rough Bergomi's k(s) = rho eta sqrt(2H) s^(H-1/2), H < 1/2.
        """
        raise NotImplementedError("Theorem 2 kernel not implemented for this model.")

    def _clone_with_params(self, **updates):
        """Return a new model instance with updated params."""
        params = self.params.copy()
        params.update(updates)
        return self.__class__(s0=self.s0, xi0=self.xi0, params=params)

    def _is_xi0_flat(self) -> bool:
        """Check if the forward variance curve xi0 is flat."""
        t_test = np.linspace(1e-10, 10, 1000)
        return np.allclose(self.xi0(t_test), self.xi0_0)

    def _is_xi0_positive(self):
        """Check if the forward variance curve xi0 is positive for t >= 0."""
        t_test = np.linspace(1e-10, 10, 1000)
        if not np.all(self.xi0(t_test) > np.array([0.0])):
            raise ValueError("xi0 must be positive for all t >= 0.")

    def impvol(self, k, T, **kwargs) -> float | np.ndarray:
        """Compute the implied volatility for log-moneyness k and maturity T."""
        raise NotImplementedError("Implied volatility not implemented for this model.")

    def total_impvar(self, k, T, **kwargs) -> float | np.ndarray:
        """Compute the total implied variance for log-moneyness k and maturity T."""
        return self.impvol(k, T, **kwargs) ** 2 * T

    def atm_skew_fd(self, T, eps=1e-4, **kwargs) -> float | np.ndarray:
        """
        Compute the ATM skew via finite differences.

        Parameters
        ----------
        T : array_like
            Time to maturity.
        eps : float
            Finite difference perturbation.

        Returns
        -------
        array_like
            ATM skew.
        """
        T = np.atleast_1d(T)
        k_pm = np.array([-eps, eps])
        impvols_pm = np.asarray([self.impvol(k_pm, Ti, **kwargs) for Ti in T])
        return (impvols_pm[:, 1] - impvols_pm[:, 0]) / (2 * eps)

    ####################################################################################
    # Monte Carlo simulation methods
    ####################################################################################

    def simulate_mc(
        self,
        tab_t: np.ndarray,
        n_mc: int,
        n_loop: int,
        seed: int | None,
        conditioning: bool,
        eps_ssr: float,
        eval_fukasawa: bool = False,
    ) -> dict:
        """
        Simulate Monte Carlo sample paths of the forward variance model.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        n_mc : int
            Total number of Monte Carlo paths to simulate.
        n_loop : int, optional
            Number of loops to split the simulation into (for memory efficiency).
            Must divide n_mc exactly. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        conditioning : bool, optional
            If True, simulate under the conditional law given the Brownian motion
            driving the spot. See Bergomi's book, Chapter 8, Appendix A.
        eps_ssr: float, optional
            If != 0, also evaluate the Skew Stickiness Ratio (SSR).
            Default is 0.0 (i.e., do not evaluate SSR).
        eval_fukasawa : bool, optional
            If True, also accumulate the kernel-weighted integrals
            `int sqrt(V_s) k(s) dB^1_s` and `int V_s k(s) ds` needed for the exact
            SSR representation formula of Theorem 2 in Fukasawa (2026), "On the Skew
            Stickiness Ratio" (arXiv:2602.05241). Default is False.

        Returns
        -------
        dict
            Dictionary with the following keys:
                - 'int_v_dt': np.ndarray, time-integrated variance for each path
                    (shape: n_mc,)
                - 'int_sqrt_v_dw': np.ndarray, stochastic integral for each path
                    (shape: n_mc,)
                - 'int_sqrt_v_k_dw', 'int_v_k_dt': np.ndarray, kernel-weighted
                    integrals for each path (shape: n_mc,), only if `eval_fukasawa`
                    is True.
        """
        raise NotImplementedError(
            "Monte Carlo simulation not implemented for this model."
        )

    def impvol_mc(
        self,
        k: float | np.ndarray,
        T: float,
        n_mc: int,
        n_disc: int,
        n_loop: int = 1,
        seed=None,
        conditioning: bool = False,
        return_skew: bool = False,
        rho_cond: float | None = None,
        eps_ssr: float = 0.0,
    ):
        """
        Estimate the implied volatility (and optionally its skew) at a given
        log-moneyness and maturity using Monte Carlo simulation.

        Parameters
        ----------
        k : float or np.ndarray
            Log-moneyness (typically 0 for ATM).
        T : float or np.ndarray
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        conditioning : bool, optional
            If True, simulate under the conditional law given the Brownian motion
            driving the spot. Default is False.
        return_skew : bool, optional
            If True, also return the estimated implied volatility skew. Default is
            False.
        eps_ssr : float, optional
            If != 0, also evaluate the Skew Stickiness Ratio (SSR) using finite
            differences. Default is 0.0 (i.e., do not evaluate SSR).

        Returns
        -------
        float or np.ndarray
            Estimated implied volatility (and optionally skew) for the given
            log-moneyness and maturity.
        """

        tab_t = np.linspace(0.0, T, n_disc + 1)
        paths = self.simulate_mc(
            tab_t=tab_t,
            n_mc=n_mc,
            n_loop=n_loop,
            seed=seed,
            conditioning=conditioning,
            eps_ssr=eps_ssr,
        )

        return implied_vol_from_paths(
            k=k,
            T=T,
            int_v_dt=paths["int_v_dt"],
            int_sqrt_v_dw=paths["int_sqrt_v_dw"],
            s0=self.s0,
            conditioning=conditioning,
            return_skew=return_skew,
            rho_cond=rho_cond,
        )

    ####################################################################################
    # Characteristic function
    ####################################################################################

    def charfunc(self, u, T) -> complex:
        """
        Characteristic function of log(S_T/S_0) under the forward variance model.
        """
        raise NotImplementedError(
            "Characteristic function not implemented for this model."
        )

    def impvol_cf(self, k, T, **kwargs) -> float | np.ndarray:
        """
        Calculate implied volatility in the Heston model using the characteristic
        function.

        Parameters
        ----------
        k : array_like
            Log strike k = log(K/S_0)
        T : float
            Maturity

        Returns
        -------
        array_like
            Black implied volatility
        """
        k = np.atleast_1d(k)
        otm_price = lewis_formula_otm_price(
            lambda u, T: self.charfunc(u=u, T=T, **kwargs), k=k, T=T
        )
        opttype = 2 * (k > 0) - 1  # OTM options
        return black_impvol(K=np.exp(k), T=T, F=1.0, value=otm_price, opttype=opttype)

    def atm_skew_cf(self, T, **kwargs):
        """
        Compute the ATM skew using the characteristic function.

        Parameters
        ----------
        T : array_like
            Time to maturity.

        Returns
        -------
        array_like
            ATM skew.
        """

        def _atm_skew_single(Ti):
            w0 = np.asarray(self.total_impvar(k=0, T=Ti, **kwargs)).item()
            integral = quad(
                lambda u: (
                    u
                    * np.asarray(
                        np.imag(self.charfunc(u=u - 0.5j, T=Ti, **kwargs))
                    ).item()
                    / (u**2 + 0.25)
                ),
                0.0,
                np.inf,
            )[0]
            return -np.exp(w0 / 8) * np.sqrt(2 / np.pi) / Ti**0.5 * integral

        return np.array([_atm_skew_single(Ti) for Ti in np.atleast_1d(T)])

    def sig_varswap(self, T, n_quad: int = 0) -> float | np.ndarray:
        """
        Compute the implied volatility of a variance swap via `scipy.integrate.quad` or
        Gauss-Legendre quadrature.

        It is given by `sqrt(1/T int_0^T xi_0^t dt)` with `xi0` the instantaneous
        forward variance.

        Parameters
        ----------
        T : array_like
            Time to maturity.
        n_quad : int, optional
            Number of quadrature points for Gauss-Legendre quadrature.
            If 0, uses `scipy.integrate.quad`.

        Returns
        -------
        array_like
            Variance swap implied volatility for each maturity in T.
        """
        if n_quad < 0:
            raise ValueError("n_quad must be non-negative.")

        T = np.atleast_1d(T)
        if n_quad > 0:
            x_leg, w_leg = gauss_legendre(0.0, 1.0, n_quad)
            out = [np.sum(w_leg * self.xi0(Ti * x_leg)) for Ti in T]
        else:
            out = [quad(lambda u, Ti=Ti: self.xi0(Ti * u), 0, 1)[0] for Ti in T]
        return np.asarray(out) ** 0.5

    ####################################################################################
    # Bergomi-Guyon expansion components
    ####################################################################################

    def mu_bg(self, t, u) -> float | np.ndarray:
        """mu(t, u) function for the Bergomi-Guyon expansion."""
        raise NotImplementedError("mu_bg is not implemented for this model.")

    def nu_bg(self, t, u, v) -> float | np.ndarray:
        """nu(t, u, v) function for the Bergomi-Guyon expansion."""
        raise NotImplementedError("nu_bg is not implemented for this model.")

    def _c_x_xi_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        C^{x,xi} functional for the Bergomi-Guyon expansion using Gauss quadrature.
        """
        raise NotImplementedError("Gauss quadrature implementation not yet available.")

    def _c_x_xi_bg_gauss_leg_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        C^{x,xi} functional for the Bergomi-Guyon expansion using Gauss-Legendre
        quadrature.
        """
        validate_positive_n_quad(n_quad)
        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis, np.newaxis]
        x_leg, w_leg = gauss_legendre(0.0, 1.0, n_quad)
        u_grid, s_grid = np.meshgrid(x_leg, x_leg, indexing="ij")
        u_grid = u_grid[np.newaxis, :, :]
        s_grid = s_grid[np.newaxis, :, :]
        w_u = w_leg[np.newaxis, :, np.newaxis]
        w_s = w_leg[np.newaxis, np.newaxis, :]
        out = np.sum(
            w_u * w_s * u_grid * self.mu_bg(T_grid * u_grid * s_grid, T_grid * u_grid),
            axis=(1, 2),
        )
        return T**2 * out

    def _c_x_xi_bg_scipy_quad(self, T: float | np.ndarray) -> float | np.ndarray:
        """
        C^{x,xi} functional for the Bergomi-Guyon expansion via `scipy.integrate.quad`.
        Note: this is very slow and not recommended for practical use but can be used
        for testing against the Gauss quadrature implementation.
        """

        def _integrand(Ti):
            out = Ti**2 * quad_complex(
                lambda u: quad_complex(lambda s: self.mu_bg(Ti * s, Ti * u), 0, u),
                0,
                1,
            )
            return np.real_if_close(out)

        return np.array([_integrand(Ti) for Ti in np.atleast_1d(T)])

    def _dxi_c_x_xi_bg_gauss_quad(self, T, t, u, n_quad) -> float | np.ndarray:
        """
        Fréchet derivative dC_t^{x,xi}(T,y)/dy^u evaluated at y=xi0.
        This is used in the calculation of c_mu_bg.
        """
        raise NotImplementedError(
            "Fréchet derivative for c_mu_bg not implemented for this model."
        )

    def _c_xi_xi_bg_gauss_leg_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        C^{xi,xi} functional for the Bergomi-Guyon expansion using Gauss-Legendre
        quadrature.
        """
        validate_positive_n_quad(n_quad)
        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis, np.newaxis]
        x_s, w_s = gauss_legendre(0.0, 1.0, n_quad)
        s_grid, u_grid, v_grid = np.meshgrid(x_s, x_s, x_s, indexing="ij")
        u_grid = u_grid[np.newaxis, :, :, :]
        s_grid = s_grid[np.newaxis, :, :, :]
        v_grid = v_grid[np.newaxis, :, :, :]
        w_s = w_s[np.newaxis, :, np.newaxis, np.newaxis]
        w_u = w_s[np.newaxis, np.newaxis, :, np.newaxis]
        w_v = w_s[np.newaxis, np.newaxis, np.newaxis, :]
        out = np.sum(
            w_s
            * w_u
            * w_v
            * (1 - s_grid) ** 2
            * self.nu_bg(
                T_grid * s_grid,
                T_grid * ((1 - s_grid) * u_grid + s_grid),
                T_grid * ((1 - s_grid) * v_grid + s_grid),
            ),
            axis=(1, 2, 3),
        )
        return T**3 * out

    def _c_mu_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        C^{mu} functional for the Bergomi-Guyon expansion using Gauss quadrature.
        """
        validate_positive_n_quad(n_quad)
        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis, np.newaxis]
        x_leg, w_leg = gauss_legendre(0.0, 1.0, n_quad)
        u_grid, s_grid = np.meshgrid(x_leg, x_leg, indexing="ij")
        u_grid = u_grid[np.newaxis, :, :]
        s_grid = s_grid[np.newaxis, :, :]
        w_u = w_leg[np.newaxis, :, np.newaxis]
        w_s = w_leg[np.newaxis, np.newaxis, :]

        out = np.sum(
            w_u
            * w_s
            * u_grid
            * self.mu_bg(t=T_grid * u_grid * s_grid, u=T_grid * u_grid)
            * self._dxi_c_x_xi_bg_gauss_quad(
                T_grid, T_grid * u_grid * s_grid, T_grid * u_grid, n_quad=n_quad
            ),
            axis=(1, 2),
        )
        return T**3 * out

    def _dxi_c_x_xi_beta_2_gauss_quad(self, T, n_quad: int) -> float | np.ndarray:
        """
        Fréchet derivative of C^{x,xi} with respect to xi0 in the direction dxi.
        This is used in the calculation of beta_2.
        """
        raise NotImplementedError("Fréchet derivative for beta_2 not implemented.")

    def _beta_1_scipy_quad(self, T) -> float | np.ndarray:
        """
        First-order beta coefficient via `scipy.integrate.quad`.

        It corresponds to the regression coefficient of
        the VS implied volatility against the log-spot:
            beta_1(T) = d<logS, sig_vs(T)> / d<logS, logS>
                      = 1/(2 sig_vs(T)) 1/T int_0^T mu_bg(0, u) rho(0) du / xi0(0)
        """

        def _integrand(Ti):
            out = quad_complex(lambda u: self.mu_bg(0, Ti * u) / self.xi0_0, 0, 1)
            out /= 2.0 * self.sig_varswap(Ti)
            return np.real_if_close(out).item()

        return np.array([_integrand(Ti) for Ti in np.atleast_1d(T)])

    def _beta_1_gauss_quad(self, T, n_quad: int) -> float | np.ndarray:
        raise NotImplementedError(
            "First-order beta coefficient via Gauss quadrature "
            "not implemented for this model."
        )

    def beta_1(self, T) -> float | np.ndarray:
        raise NotImplementedError(
            "First-order beta coefficient not implemented for this model."
        )

    def _beta_2_gauss_quad(self, T, n_quad: int) -> float | np.ndarray:
        _sig_vs = self.sig_varswap(T, n_quad)
        beta_2 = self._dxi_c_x_xi_beta_2_gauss_quad(T, n_quad) * _sig_vs
        beta_2 -= self._c_x_xi_bg_gauss_quad(T, n_quad) * self._beta_1_gauss_quad(
            T, n_quad
        )
        beta_2 /= 4.0 * T * _sig_vs**2
        return beta_2

    def beta_2(self, T, n_quad: int) -> float | np.ndarray:
        raise NotImplementedError(
            "Second-order beta coefficient not implemented for this model."
        )

    def ssr_mc_all(
        self,
        T: float | np.ndarray,
        n_mc: int,
        n_disc: int,
        eps_ssr: float,
        n_loop: int = 1,
        seed=None,
        conditioning: bool = False,
        n_quad: int = 20,
        rho_cond: float | None = None,
        n_batch: int = 1,
    ):
        """
        Estimate SSR, VS SSR, beta, and ATM skew using Monte Carlo simulation.

        Parameters
        ----------
        T : float
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        eps_ssr : float
            Perturbation size for finite difference estimation of the Skew Stickiness
            Ratio (SSR). Must be nonzero.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        conditioning : bool, optional
            If True, simulate under the conditional law given the Brownian motion
            driving the spot. Default is False.
        n_quad : int, optional
            Number of quadrature points for the variance swap rate estimation.
            Default is 20.
        n_batch : int, optional
            Number of independent Monte Carlo batches. If greater than 1, returns
            batch-mean estimates and 95% confidence intervals. Default is 1.

        Returns
        -------
        dict
            Monte Carlo estimates for SSR, VS SSR, beta, and ATM skew. If `n_batch`
            is greater than 1, also includes 95% confidence intervals.
        """
        if eps_ssr == 0.0:
            raise ValueError(
                "eps_ssr must be nonzero for finite difference SSR estimation."
            )

        n_batch = int(n_batch)
        if n_batch <= 0:
            raise ValueError("n_batch must be a positive integer.")

        n_loop = int(n_loop)
        if n_loop <= 0:
            raise ValueError("n_loop must be a positive integer.")

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        n_mc_loop = int(n_mc_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")

        T = np.atleast_1d(T)

        print("\nComputing SSR:")
        print("----------------")
        print(f"Model: {self.__class__.__name__}")
        print("Parameters:")
        print("Maturity T: ", T)
        print(f"Monte Carlo paths: {n_mc}")
        print(f"Number of time steps: {n_disc}")
        print(f"Finite difference perturbation for SSR: {eps_ssr}")
        print(f"Number of simulation loops: {n_loop}")
        print(f"Number of independent batches: {n_batch}\n")

        def _atm_vol(Ti, int_v_dt, int_sqrt_v_dw, return_skew=False):
            return _atm_impvol_from_path_integrals(
                T=Ti,
                int_v_dt=int_v_dt,
                int_sqrt_v_dw=int_sqrt_v_dw,
                s0=self.s0,
                conditioning=conditioning,
                rho_cond=rho_cond,
                return_skew=return_skew,
            )

        def _ssr_atm_skew_single(Ti, batch_seed):

            paths = self.simulate_mc(
                tab_t=np.linspace(0.0, Ti, n_disc + 1),
                n_mc=n_mc,
                n_loop=n_loop,
                seed=batch_seed,
                conditioning=conditioning,
                eps_ssr=eps_ssr,
            )

            atm_impvol_i, atm_skew_i = _atm_vol(
                Ti, paths["int_v_dt"], paths["int_sqrt_v_dw"], return_skew=True
            )

            if atm_skew_i == 0.0:
                raise ValueError("Estimated ATM skew is zero, cannot compute SSR.")

            atm_impvol_shifted_i = _atm_vol(
                Ti,
                paths["int_v_dt_shifted"],
                paths["int_sqrt_v_dw_shifted"],
                return_skew=False,
            )

            atm_impvol_shifted_i = np.asarray(atm_impvol_shifted_i)
            ssr_fd_i = (atm_impvol_shifted_i - atm_impvol_i) / (eps_ssr * atm_skew_i)
            return ssr_fd_i.item(), atm_skew_i.item()

        n_expiries = len(T)

        def _ssr_atm_skew_with_progress(i, Ti, batch_seed):
            print(f"{i}/{n_expiries}: expiry {Ti:.4f}")
            return _ssr_atm_skew_single(Ti, batch_seed)

        beta_1 = self._beta_1_gauss_quad(T, n_quad=n_quad)

        def _run_batch(batch_seed):
            ssr_fd, atm_skew = np.array(
                [
                    _ssr_atm_skew_with_progress(i, Ti, batch_seed)
                    for i, Ti in enumerate(T, start=1)
                ]
            ).T

            ssr_vs = beta_1 / atm_skew
            beta = ssr_fd * atm_skew

            return {
                "ssr_fd": ssr_fd,
                "ssr_vs": ssr_vs,
                "beta": beta,
                "atm_skew": atm_skew,
            }

        if n_batch == 1:
            return _run_batch(seed)

        # Run multiple independent batches and aggregate results with confidence
        # intervals.
        rng = np.random.default_rng(seed)
        batch_seeds = rng.integers(0, 2**32 - 1, size=n_batch)
        out_batch = []
        for i, batch_seed in enumerate(batch_seeds, start=1):
            print(f"\nRunning batch {i}/{n_batch} with seed {batch_seed}")
            out_batch.append(_run_batch(batch_seed))

        out = {}
        for key in ("ssr_fd", "ssr_vs", "beta", "atm_skew"):
            values = np.asarray([batch_out[key] for batch_out in out_batch])
            out[key] = values.mean(axis=0)
            err = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(n_batch)
            out[f"{key}_low"] = out[key] - err
            out[f"{key}_high"] = out[key] + err

        return out

    def ssr_fukasawa_all(
        self,
        T: float | np.ndarray,
        n_mc: int,
        n_disc: int,
        n_loop: int = 1,
        seed=None,
        n_batch: int = 1,
    ):
        """
        Estimate the SSR at t=0 from the exact representation formula R_0 = X_0/Y_0
        of Theorem 2 in Fukasawa (2026), "On the Skew Stickiness Ratio"
        (arXiv:2602.05241), for Bergomi-type models (Eq. 11 in that paper).

        Unlike `ssr_mc_all`, this does not rely on finite-difference bumping of the
        forward variance curve: both X_0 and Y_0 are estimated from a single,
        un-bumped Monte Carlo simulation, using the kernel-weighted integrals
        returned by `simulate_mc(..., eval_fukasawa=True)`.

        Parameters
        ----------
        T : float or np.ndarray
            Maturity.
        n_mc : int
            Number of Monte Carlo paths.
        n_disc : int
            Number of time discretization steps.
        n_loop : int, optional
            Number of simulation loops for memory efficiency. Default is 1.
        seed : int or None, optional
            Random seed for reproducibility. Default is None.
        n_batch : int, optional
            Number of independent Monte Carlo batches. If greater than 1, returns
            batch-mean estimates and 95% confidence intervals. Default is 1.

        Returns
        -------
        dict
            Monte Carlo estimate for the Theorem 2 SSR (key `"ssr_fukasawa"`). If
            `n_batch` is greater than 1, also includes 95% confidence intervals.
        """
        n_batch = int(n_batch)
        if n_batch <= 0:
            raise ValueError("n_batch must be a positive integer.")

        n_loop = int(n_loop)
        if n_loop <= 0:
            raise ValueError("n_loop must be a positive integer.")

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")

        T = np.atleast_1d(T)

        print("\nComputing Fukasawa SSR:")
        print("----------------")
        print(f"Model: {self.__class__.__name__}")
        print("Maturity T: ", T)
        print(f"Monte Carlo paths: {n_mc}")
        print(f"Number of time steps: {n_disc}")
        print(f"Number of simulation loops: {n_loop}")
        print(f"Number of independent batches: {n_batch}\n")

        def _ssr_fukasawa_single(Ti, batch_seed):
            paths = self.simulate_mc(
                tab_t=np.linspace(0.0, Ti, n_disc + 1),
                n_mc=n_mc,
                n_loop=n_loop,
                seed=batch_seed,
                conditioning=False,
                eps_ssr=0.0,
                eval_fukasawa=True,
            )
            int_v_dt = paths["int_v_dt"]
            int_sqrt_v_dw = paths["int_sqrt_v_dw"]
            kernel_stoch = paths["int_sqrt_v_k_dw"]
            kernel_drift = paths["int_v_k_dt"]

            S_T = self.s0 * np.exp(-0.5 * int_v_dt + int_sqrt_v_dw)
            F = S_T.mean()

            atm_impvol = _black_atm_impvol_from_price(
                F=F, T=Ti, price=np.maximum(S_T - F, 0.0).mean()
            )
            d2 = -0.5 * atm_impvol * Ti**0.5
            digit = np.mean(S_T >= F)
            Y = stats.norm.cdf(d2) - digit
            if Y == 0.0:
                raise ValueError("Estimated Y_0 is zero, cannot compute Theorem 2 SSR.")

            indicator = S_T < F
            X = -np.mean(indicator * S_T * (kernel_stoch - kernel_drift)) / (
                2.0 * F * self.xi0_0**0.5
            )

            return (X / Y).item()

        n_expiries = len(T)

        def _ssr_fukasawa_with_progress(i, Ti, batch_seed):
            print(f"{i}/{n_expiries}: expiry {Ti:.4f}")
            return _ssr_fukasawa_single(Ti, batch_seed)

        def _run_batch(batch_seed):
            ssr_fukasawa = np.array(
                [
                    _ssr_fukasawa_with_progress(i, Ti, batch_seed)
                    for i, Ti in enumerate(T, start=1)
                ]
            )
            return {"ssr_fukasawa": ssr_fukasawa}

        if n_batch == 1:
            return _run_batch(seed)

        rng = np.random.default_rng(seed)
        batch_seeds = rng.integers(0, 2**32 - 1, size=n_batch)
        out_batch = []
        for i, batch_seed in enumerate(batch_seeds, start=1):
            print(f"\nRunning batch {i}/{n_batch} with seed {batch_seed}")
            out_batch.append(_run_batch(batch_seed))

        out = {}
        for key in ("ssr_fukasawa",):
            values = np.asarray([batch_out[key] for batch_out in out_batch])
            out[key] = values.mean(axis=0)
            err = 1.96 * values.std(axis=0, ddof=1) / np.sqrt(n_batch)
            out[f"{key}_low"] = out[key] - err
            out[f"{key}_high"] = out[key] + err

        return out

    def ssr_bg_all(self, T: float | np.ndarray, n_quad: int, **kwargs) -> dict:
        """
        Compute SSR, VS SSR, beta, and ATM skew from the Bergomi-Guyon expansion
        up to second order.
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))
        _sig_vs = self.sig_varswap(T, n_quad=n_quad)
        _c_x_xi_bg = self._c_x_xi_bg_gauss_quad(T, n_quad=n_quad)
        beta_1 = self._beta_1_gauss_quad(T, n_quad=n_quad)
        return self._ssr_bg_all_from_quadrature_terms(
            T=T,
            n_quad=n_quad,
            sig_vs=_sig_vs,
            c_x_xi_bg=_c_x_xi_bg,
            beta_1=beta_1,
        )

    def _ssr_bg_all_from_quadrature_terms(
        self,
        T: float | np.ndarray,
        n_quad: int,
        sig_vs,
        c_x_xi_bg,
        beta_1,
        beta_2=None,
        c_mu_bg=None,
    ) -> dict:
        """Assemble BG SSR outputs from reusable quadrature terms."""
        T = np.atleast_1d(np.asarray(T, dtype=float))

        if beta_2 is None:
            beta_2 = self._dxi_c_x_xi_beta_2_gauss_quad(T, n_quad) * sig_vs
            beta_2 -= c_x_xi_bg * beta_1
            beta_2 /= 4.0 * T * sig_vs**2

        if c_mu_bg is None:
            c_mu_bg = self._c_mu_bg_gauss_quad(T, n_quad=n_quad)

        atm_skew_1 = c_x_xi_bg / (2.0 * T**2 * sig_vs**3)
        atm_skew_2 = (4 * c_mu_bg * T * sig_vs**2 - 3 * c_x_xi_bg**2) / (
            8 * T**3 * sig_vs**5
        )

        return self._ssr_bg_all_from_expansion_terms(
            beta_1=beta_1,
            beta_2=beta_2,
            atm_skew_1=atm_skew_1,
            atm_skew_2=atm_skew_2,
        )

    @staticmethod
    def _ssr_bg_all_from_expansion_terms(
        beta_1,
        beta_2,
        atm_skew_1,
        atm_skew_2,
    ) -> dict:
        """Assemble BG SSR outputs from beta and ATM skew expansion terms."""
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
