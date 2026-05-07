from collections import OrderedDict
from collections.abc import Callable
from functools import lru_cache
import hashlib

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad_vec
from pymittagleffler import mittag_leffler

import pade
from model import (
    ForwardVarianceModel,
    require_params,
    validate_interval,
    validate_nonnegative,
    validate_positive,
    validate_positive_n_quad,
)
from utils import gauss_legendre, jacobi_quadrature


ML_TWO_ARRAY_CACHE_SIZE = 16
_ML_TWO_ARRAY_CACHE = OrderedDict()


def _as_maturity_array(T):
    """Return maturity input as at least one-dimensional array."""
    return np.atleast_1d(np.asarray(T))


@lru_cache(maxsize=4096)
def _ml_two_scalar_cached(alpha: float, dtype_str: str, data: bytes):
    z = np.frombuffer(data, dtype=np.dtype(dtype_str))[0].item()
    return mittag_leffler(z, alpha, alpha)


def _ml_two_array_key(z: np.ndarray, alpha: float):
    z = np.ascontiguousarray(z)
    digest = hashlib.blake2b(z.view(np.uint8), digest_size=16).digest()
    return (alpha, z.shape, z.dtype.str, digest), z


def clear_ml_two_cache() -> None:
    """Clear cached Mittag-Leffler values used by the rough Heston kernels."""
    _ml_two_scalar_cached.cache_clear()
    _ML_TWO_ARRAY_CACHE.clear()


def ml_two_cache_info() -> dict:
    """Return cache diagnostics for rough Heston Mittag-Leffler evaluations."""
    return {
        "scalar": _ml_two_scalar_cached.cache_info(),
        "array_size": len(_ML_TWO_ARRAY_CACHE),
        "array_maxsize": ML_TWO_ARRAY_CACHE_SIZE,
    }


class RoughHestonModel(ForwardVarianceModel):
    """Rough Heston model."""

    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray],
        s0: float = 1.0,
    ) -> None:
        """
        Initialize Rough Heston model.

        Parameters
        ----------
        xi0 : callable
            Initial forward variance curve function.
        params : dict
            Dictionary containing model parameters:
            - eta: volatility of volatility
            - H: Hurst parameter
            - kappa: mean reversion speed
            - rho: correlation between Brownian motions
        s0 : float, optional
            Initial stock price, by default 1.0
        """
        self.eta, self.H, self.kappa, self.rho = require_params(
            params, ("eta", "H", "kappa", "rho")
        )

        super().__init__(params=params, xi0=xi0, s0=s0)
        self._check_params()
        self.alpha = self.H + 0.5

    def _check_params(self):
        """Validate Rough Heston parameters."""
        validate_positive("eta", self.eta)
        validate_nonnegative("kappa", self.kappa)
        validate_interval("rho", self.rho, -1, 1)
        validate_interval("H", self.H, 0, 1, closed=False)

    def lbd_xi(self, u, t):
        return (
            self.eta
            * (u - t) ** (self.H - 0.5)
            * ml_two(-self.kappa * (u - t) ** self.alpha, self.alpha)
            * self.xi0(t) ** 0.5
        )

    def charfunc(self, u, T, n_pade: int, n_quad: int):
        """
        Padé-based rational approximation of the rough Heston characteristic function.

        Uses a Padé approximant for the integrand and Gauss-Legendre (or quad_vec)
        integration to compute E[exp(i u X_T)] for given u and T.

        Parameters
        ----------
        u : float or ndarray
            Fourier argument(s) for the characteristic function.
        T : float or ndarray
            Time(s) to maturity.
        n_pade : int
            Order of Padé approximation.
        n_quad : int
            Number of quadrature points for integration.

        Returns
        -------
        ndarray
            Characteristic function values for each tau.
        """
        pade.validate_pade_order(n_pade)

        T = _as_maturity_array(T)
        if n_quad > 0:
            # Gauss quadrature
            x_le, w_le = gauss_legendre(0, 1, n_quad)
            T_grid = T[:, None]
            log_charfunc = (
                w_le[None, :]
                * self.xi0(T_grid * (1 - x_le[None, :]))
                * pade.g(
                    a=u,
                    t=T_grid * x_le[None, :],
                    n_pade=n_pade,
                    params=self.params,
                )
            )
            log_charfunc = np.sum(log_charfunc, axis=1)
        else:
            # Using scipy.quad_vec() is longer.
            log_charfunc = quad_vec(
                lambda t: (
                    self.xi0(T * (1 - t))
                    * pade.g(
                        a=u,
                        t=T * t,
                        n_pade=n_pade,
                        params=self.params,
                    )
                ),
                0,
                1,
                epsrel=1e-10,
                limit=1000,
            )[0]
        charfunc = np.exp(T * log_charfunc)
        return charfunc

    def _charfunc_from_gauss_context(
        self,
        u,
        T: float,
        n_pade: int,
        t_grid: np.ndarray,
        weighted_xi0: np.ndarray,
    ):
        log_charfunc = np.sum(
            weighted_xi0
            * pade.g(
                a=u,
                t=t_grid,
                n_pade=n_pade,
                params=self.params,
            )
        )
        return np.exp(T * log_charfunc)

    def impvol(self, k, T, n_pade: int, n_quad: int):
        """
        Calculate implied volatility in the rough Heston model using a rational
        approximation of the characteristic function.

        Parameters
        ----------
        k : float
            Log strike
        T : float
            Time to maturity
        params : dict
            Model parameters
        xi : ndarray
            Volatility curve values
        n : int
            Order of rational approximation

        Returns
        -------
        float
            Black implied volatility
        """
        pade.validate_pade_order(n_pade)

        return self.impvol_cf(k, T, n_pade=n_pade, n_quad=n_quad)

    def atm_skew(self, T, **kwargs):
        return self.atm_skew_cf(T, **kwargs)

    def xi0_shifted(self, u, eps):
        """Shifted initial forward variance curve for finite difference SSR."""
        if self.H == 0.5:
            raise NotImplementedError("Heston case (H=1/2) not implemented yet")
        else:
            return np.where(
                u == 0.0,
                self.xi0_0,
                self.xi0(u)
                + eps
                * self.eta
                * self.rho
                * u ** (self.H - 0.5)
                * ml_two(-self.kappa * u**self.alpha, self.alpha),
            )

    def ssr_fd(self, T, n_pade, n_quad, eps: float):
        """
        Compute Skew-Stickiness Ratio (SSR) via finite difference.
        """
        rheston_shift = self._clone_with_params()
        rheston_shift.xi0 = lambda u: self.xi0_shifted(u, eps)

        def _ssr_fd_single(Ti):
            out = (
                rheston_shift.impvol(0.0, Ti, n_pade, n_quad)
                - self.impvol(0.0, Ti, n_pade, n_quad)
            ) / eps
            out /= self.atm_skew_cf(T=Ti, n_pade=n_pade, n_quad=n_quad)
            return out[0]

        return np.array([_ssr_fd_single(Ti) for Ti in _as_maturity_array(T)])

    def ssr_vs(self, T, n_pade: int, n_quad: int, eps: float = 0.0):
        validate_positive_n_quad(n_quad)

        beta_1 = self._beta_1_gauss_quad(T, n_quad)

        return beta_1 / self.atm_skew_cf(T=T, n_pade=n_pade, n_quad=n_quad)

    def _beta_1_gauss_quad(self, T, n_quad: int):
        """First-order beta coefficient using Gauss quadrature."""
        validate_positive_n_quad(n_quad)
        T = _as_maturity_array(T)
        if self.kappa == 0:
            integral = 1 / gamma(self.H + 0.5)
        else:
            x_leg, w_leg = gauss_legendre(0.0, 1.0, n_quad)
            integral = np.sum(
                w_leg[None, :]
                * ml_two(
                    -self.kappa * T[:, None] ** self.alpha * x_leg[None, :], self.alpha
                ),
                axis=1,
            )
        integral *= (
            self.rho
            * self.eta
            * T ** (self.H - 0.5)
            / ((2 * self.H + 1) * self.sig_varswap(T, n_quad))
        )
        return integral

    def mu_bg(self, t, u):
        return (
            self.rho
            * self.eta
            * (u - t) ** (self.H - 0.5)
            * self.xi0(t)
            * ml_two(-self.kappa * (u - t) ** self.alpha, self.alpha)
        )

    def nu_bg(self, t, u, v) -> float | np.ndarray:
        return (
            self.eta**2
            * self.xi0(t)
            * ((u - t) * (v - t)) ** (self.H - 0.5)
            * ml_two(-self.kappa * (u - t) ** self.alpha, self.alpha)
            * ml_two(-self.kappa * (v - t) ** self.alpha, self.alpha)
        )

    def _c_x_xi_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)

        T = _as_maturity_array(T)
        s_grid, w_s = gauss_legendre(0.0, 1.0, n_quad)
        v_grid, w_v = jacobi_quadrature(n_quad, 0.0, self.H - 0.5)
        s_grid, v_grid = np.meshgrid(s_grid, v_grid, indexing="ij")
        s_grid = s_grid[np.newaxis, :, :]
        v_grid = v_grid[np.newaxis, :, :]
        w_s = w_s[np.newaxis, :, np.newaxis]
        w_v = w_v[np.newaxis, np.newaxis, :]
        T_grid = T[:, np.newaxis, np.newaxis]

        ml_grid = ml_two(
            -self.kappa
            * (0.5 * T_grid * (1 - s_grid) * (1 + v_grid)) ** (self.H + 0.5),
            self.alpha,
        )
        out = np.sum(
            w_s
            * w_v
            * self.xi0(T_grid * s_grid)
            * (1 - s_grid) ** (self.H + 0.5)
            * ml_grid,
            axis=(1, 2),
        )
        out *= self.rho * self.eta * T ** (self.H + 1.5) / 2 ** (self.H + 0.5)
        return out

    def _c_mu_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)

        T = _as_maturity_array(T)

        # s in [0,1]
        x_s, w_s = gauss_legendre(0.0, 1.0, n_quad)
        # u in [-1,1] with weight (1-u)^(H+1/2) (1+u)^(H-1/2)
        x_u, w_u = jacobi_quadrature(n_quad, self.H + 0.5, self.H - 0.5)
        # v in [-1,1] with weight (1+v)^(H-1/2)
        x_v, w_v = jacobi_quadrature(n_quad, 0.0, self.H - 0.5)

        # vectorize the integrand over the 3D grid of (s,u,v) points
        s_grid, u_grid, v_grid = np.meshgrid(x_s, x_u, x_v, indexing="ij")
        s_grid = s_grid[np.newaxis, :, :, :]
        u_grid = u_grid[np.newaxis, :, :, :]
        v_grid = v_grid[np.newaxis, :, :, :]
        w_s = w_s[np.newaxis, :, np.newaxis, np.newaxis]
        w_u = w_u[np.newaxis, np.newaxis, :, np.newaxis]
        w_v = w_v[np.newaxis, np.newaxis, np.newaxis, :]
        T_grid = T[:, np.newaxis, np.newaxis, np.newaxis]

        # The integrand is the product of the xi0 term, the kernel terms, and the
        # weights.
        ml_u = ml_two(
            -self.kappa
            * (0.5 * T_grid * (1 - s_grid) * (1 + u_grid)) ** (self.H + 0.5),
            self.alpha,
        )
        ml_v = ml_two(
            -self.kappa
            * (0.25 * T_grid * (1 - s_grid) * (1 - u_grid) * (1 + v_grid))
            ** (self.H + 0.5),
            self.alpha,
        )
        out = np.sum(
            w_s
            * w_u
            * w_v
            * self.xi0(T_grid * s_grid)
            * (1 - s_grid) ** (2 * self.H + 1)
            * ml_u
            * ml_v,
            axis=(1, 2, 3),
        )
        out *= (
            self.eta**2 * self.rho**2 * T ** (2 + 2 * self.H) / 2 ** (3 * self.H + 1.5)
        )

        return out

    def _dxi_c_x_xi_beta_2_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        Fréchet derivative of C^{x,xi} with respect to xi0 in the direction dxi.
        This is used in the calculation of beta_2_bg.
        """
        validate_positive_n_quad(n_quad)

        T = _as_maturity_array(T)
        x_grid, w_x = jacobi_quadrature(n_quad, self.H + 0.5, self.H - 0.5)
        u_grid, w_u = jacobi_quadrature(n_quad, 0.0, self.H - 0.5)
        x_grid, u_grid = np.meshgrid(x_grid, u_grid, indexing="ij")

        x_grid = x_grid[np.newaxis, :, :]
        u_grid = u_grid[np.newaxis, :, :]
        w_x = w_x[np.newaxis, :, np.newaxis]
        w_u = w_u[np.newaxis, np.newaxis, :]
        T_grid = T[:, np.newaxis, np.newaxis]

        ml_x = ml_two(
            -self.kappa * (0.5 * T_grid * (1 + x_grid)) ** (self.H + 0.5), self.alpha
        )
        ml_u = ml_two(
            -self.kappa
            * (0.25 * T_grid * (1 - x_grid) * (1 + u_grid)) ** (self.H + 0.5),
            self.alpha,
        )
        out = np.sum(w_x * w_u * ml_x * ml_u, axis=(1, 2))
        out *= (
            self.eta**2 * self.rho**2 * T ** (2 * self.H + 1) / 2 ** (3 * self.H + 1.5)
        )
        return out

    def ssr_cf(self, T, n_pade: int, n_quad: int):
        """
        SSR via characteristic-function integrals using a Padé approx.

        Parameters
        ----------
        T : array_like
            Maturities.
        params : dict
            Model parameters (expects at least 'H', 'rho', 'eta', 'kappa' when needed).
        xi_curve : callable
            Forward variance curve xi_curve(t).
        n_pade : int, optional
            Padé order (default 2).
        n_quad : int, optional
            Quadrature points for inner charfunc integration (default 30).

        Returns
        -------
        ndarray
            SSR values matching shape of T.
        """
        T = _as_maturity_array(T)
        ssr = np.zeros_like(T)

        if self.H == 0.5:
            # Heston case (H=1/2) - convert to Heston variables
            raise NotImplementedError("Heston case (H=1/2) not implemented yet")
        else:
            for i, T_i in enumerate(T):
                if n_quad > 0:
                    x_leg, w_leg = gauss_legendre(0.0, 1.0, n_quad)
                    t_grid = T_i * x_leg
                    weighted_xi0 = w_leg * self.xi0(T_i * (1.0 - x_leg))

                def integrand(a, T_i=T_i):
                    a_shifted = a - 0.5j
                    if n_quad > 0:
                        cf = self._charfunc_from_gauss_context(
                            u=a_shifted,
                            T=T_i,
                            n_pade=n_pade,
                            t_grid=t_grid,
                            weighted_xi0=weighted_xi0,
                        )
                    else:
                        cf = self.charfunc(
                            u=a_shifted,
                            T=T_i,
                            n_pade=n_pade,
                            n_quad=n_quad,
                        )[0]
                    h_pade = pade._h_pade(
                        tau=T_i, a=a_shifted, params=self.params, n_pade=n_pade
                    )
                    denom = a**2 + 0.25
                    return np.array(
                        [
                            np.real(self.rho * self.eta * h_pade * cf / denom),
                            np.imag(cf * a / denom),
                        ]
                    )

                integral = quad_vec(integrand, 0, np.inf)[0]
                ssr[i] = integral[0] / integral[1]

        return ssr

    def ssr_all(self, T, n_pade, n_quad):
        """Compute SSR, VS SSR, beta, and ATM skew."""
        # precompute common quantities
        atm_skew = self.atm_skew_cf(T, n_pade=n_pade, n_quad=n_quad)
        ssr_vs = self._beta_1_gauss_quad(T, n_quad=n_quad) / atm_skew
        ssr_cf = self.ssr_cf(T, n_pade=n_pade, n_quad=n_quad)
        beta = ssr_cf * atm_skew

        return {
            "ssr_cf": ssr_cf,
            "ssr_vs": ssr_vs,
            "beta": beta,
            "atm_skew": atm_skew,
        }

    def ssr_forest(self, T, explicit=False):
        """
        Compute SSR via the Forest expansion.

        Parameters
        ----------
        T : float or np.ndarray
            Maturity horizon(s) (T > 0).
        explicit : bool, optional
            If True use explicit coefficient formulas (default False).

        Returns
        -------
        float or np.ndarray
            SSR value(s) with the same shape as T.

        Notes
        -----
        This implementation assumes a flat initial forward variance curve and zero speed
        of mean reversion (kappa = 0).

        References
        ----------
        Friz, Gatheral (2025), "Computing the SSR". Proposition 5.8.
        """
        rho = self.rho
        eta = self.eta
        xi0 = self.xi0_0
        xi0_T = xi0 * T

        # --- gamma function shorthands ---
        gamma1_alpha = gamma(1 + self.alpha)
        gamma2_alpha = gamma(2 + self.alpha)
        gamma1_2alpha = gamma(1 + 2 * self.alpha)
        gamma2_2alpha = gamma(2 + 2 * self.alpha)
        gamma1_3alpha = gamma(1 + 3 * self.alpha)
        gamma2_3alpha = gamma(2 + 3 * self.alpha)

        if explicit:
            c0 = -gamma1_2alpha / (4.0 * gamma1_alpha**2 * gamma1_3alpha) + 3.0 / (
                8.0 * (2.0 * self.alpha + 1.0) * gamma1_alpha**3
            )
            c1 = (
                -1.0 / gamma1_3alpha
                + 3.0 / (2.0 * gamma2_alpha * gamma1_2alpha)
                + 3.0 / (2.0 * gamma1_alpha * gamma2_2alpha)
                - 15.0 / (8.0 * gamma1_alpha * gamma2_alpha**2)
            )
            d0 = (
                15.0 / (8.0 * (2.0 * self.alpha + 1.0) * gamma1_alpha**2 * gamma2_alpha)
                - 3.0 * gamma1_2alpha / (4.0 * gamma1_alpha**2 * gamma2_3alpha)
                - 3.0 / (2.0 * gamma1_alpha * gamma1_2alpha * (3.0 * self.alpha + 1.0))
            )
            d1 = (
                15.0 / (2.0 * gamma2_alpha * gamma2_2alpha)
                - 3.0 / gamma2_3alpha
                - 35.0 / (8.0 * gamma2_alpha**3)
            )

            numerator = (
                rho * xi0 * eta * T ** (self.alpha + 1.0) / gamma1_alpha
                + rho**2
                * xi0
                * eta**2
                * T ** (2.0 * self.alpha + 1.0)
                * (
                    1.0 / (2.0 * gamma1_2alpha)
                    - 1.0 / (4.0 * gamma1_alpha * gamma2_alpha)
                )
                + rho * eta**3 * T ** (3.0 * self.alpha) * (c0 + rho**2 * c1)
            )
            denominator = (
                xi0 * rho * eta * T ** (1.0 + self.alpha) / gamma2_alpha
                + xi0
                * rho**2
                * eta**2
                * T ** (2.0 * self.alpha + 1.0)
                * (1.0 / gamma2_2alpha - 3.0 / (4.0 * gamma2_alpha**2))
                + rho * eta**3 * T ** (3.0 * self.alpha) * (d0 + rho**2 * d1)
            )
        else:
            # --- kernel terms ---
            o_g = rho * eta * xi0 * T ** (self.alpha + 1) / gamma2_alpha
            o_o = (
                eta**2
                * xi0
                * T ** (2 * self.alpha + 1)
                / (gamma1_alpha**2 * (2 * self.alpha + 1))
            )
            o_g_g = rho**2 * eta**2 * xi0 * T ** (2 * self.alpha + 1) / gamma2_2alpha
            g_o_o = (
                rho
                * eta**3
                * xi0
                * T ** (3 * self.alpha + 1)
                / (gamma1_alpha * gamma1_2alpha * (3 * self.alpha + 1))
            )
            o_o_g = (
                rho
                * eta**3
                * gamma1_2alpha
                * xi0
                * T ** (3 * self.alpha + 1)
                / (gamma1_alpha**2 * gamma2_3alpha)
            )
            o_g_g_g = rho**3 * eta**3 * xi0 * T ** (3 * self.alpha + 1) / gamma2_3alpha

            # --- derivatives ---
            dxi_o = eta * T**self.alpha / gamma1_alpha
            dxi_o_g = rho * eta**2 * T ** (2 * self.alpha) / gamma1_2alpha
            dxi_o_o = (
                eta**3
                * gamma1_2alpha
                * T ** (3 * self.alpha)
                / (gamma1_alpha**2 * gamma1_3alpha)
            )
            dxi_o_g_g = rho**2 * eta**3 * T ** (3 * self.alpha) / gamma1_3alpha

            # --- SSR formula ---
            numerator = (
                dxi_o
                + dxi_o_g / 2.0
                - dxi_o_o / (4.0 * xi0_T)
                - dxi_o_g_g / xi0_T
                - o_g * dxi_o / (4 * xi0_T)
                + 3 * o_g * dxi_o_g / (2 * xi0_T**2)
                + 3 * dxi_o * o_o / (8 * xi0_T**2)
                + 3 * dxi_o * o_g_g / (2 * xi0_T**2)
                - 15 * dxi_o * o_g**2 / (8 * xi0_T**3)
            )
            numerator *= rho * xi0_T

            denominator = (
                o_g
                + o_g_g
                - 3 * o_g**2 / (4 * xi0_T)
                - 105 * o_g**3 / (24 * xi0_T**3)
                + 15 * o_o * o_g / (8 * xi0_T**2)
                + 15 * o_g_g * o_g / (2 * xi0_T**2)
                - 3 * o_o_g / (4 * xi0_T)
                - 3 * g_o_o / (2 * xi0_T)
                - 3 * o_g_g_g / xi0_T
            )

        return numerator / denominator

    ####################################################################################
    # flat forward variance curve and kappa = 0 case
    ####################################################################################

    def _beta_1_bg_flat_kappa_zero(self, T):
        if self.kappa != 0:
            raise ValueError("This function assumes kappa = 0.")
        return (
            0.5
            * self.eta
            * self.rho
            * T ** (self.H - 0.5)
            / (np.sqrt(self.xi0_0) * gamma(self.H + 1.5))
        )

    def _beta_2_bg_flat_kappa_zero(self, T):
        if self.kappa != 0:
            raise ValueError("This function assumes kappa = 0.")
        return (
            self._beta_1_bg_flat_kappa_zero(T)
            * 0.25
            * self.eta
            * self.rho
            * (
                2 * gamma(self.H + 1.5) / gamma(2 * self.H + 2)
                - 1 / gamma(self.H + 2.5)
            )
            * T ** (self.H + 0.5)
        )

    def _atm_skew_1_bg_flat_kappa_zero(self, T):
        if self.kappa != 0:
            raise ValueError("This function assumes kappa = 0.")
        return self._beta_1_bg_flat_kappa_zero(T) / (self.H + 1.5)

    def _atm_skew_2_bg_flat_kappa_zero(self, T):
        if self.kappa != 0:
            raise ValueError("This function assumes kappa = 0.")
        return (
            (
                self._atm_skew_1_bg_flat_kappa_zero(T)
                * self.eta
                * self.rho
                / gamma(self.H + 2.5)
            )
            * (gamma(self.H + 2.5) ** 2 / gamma(2 * self.H + 3) - 0.75)
            * T ** (self.H + 0.5)
        )

    def _ssr_0_bg_flat_kappa_zero(self):
        return self.H + 1.5

    def _ssr_1_vs_bg_flat_kappa_zero(self, T):
        if self.kappa != 0:
            raise ValueError("This function assumes kappa = 0.")
        out = -self.eta * self.rho * (self.H + 1.5) / gamma(self.H + 2.5)
        out *= (gamma(self.H + 2.5) ** 2 / gamma(2 * self.H + 3) - 0.75) * T ** (
            self.H + 0.5
        )
        return out

    def _ssr_1_bg_flat_kappa_zero(self, T):
        if self.kappa != 0:
            raise ValueError("This function assumes kappa = 0.")
        return self._ssr_1_vs_bg_flat_kappa_zero(T) + 0.25 * self.eta * self.rho * (
            self.H + 1.5
        ) * (
            2 * gamma(self.H + 1.5) / gamma(2 * self.H + 2) - 1 / gamma(self.H + 2.5)
        ) * T ** (self.H + 0.5)


def get_params_rough_heston(id: int):
    """Get Rough Heston parameters for a given id."""
    if id not in [1, 2]:
        raise ValueError("Invalid id. Please choose a valid id.")
    if id == 1:
        # Gatheral and Radoicic (2024), A generalization of the rational rough Heston
        # approximation, QF
        params = {"eta": 0.4, "H": 0.05, "kappa": 1.0, "rho": -0.65}
        xi0 = lambda t: 0.04 * np.ones_like(t)
    if id == 2:
        # Gatheral (2022), Efficient simulation of affine forward variance models, Risk
        params = {"eta": 0.8, "H": 0.05, "kappa": 0.0, "rho": -0.65}
        xi0 = lambda t: 0.025 * np.ones_like(t)
    return params, xi0


# @np.vectorize
# def ml_two(z, alpha):
#     return Mittag_Leffler_two(z, alpha, alpha)


def ml_two(z, alpha):
    alpha = float(alpha)
    z = np.asarray(z)

    if z.ndim == 0:
        z = np.ascontiguousarray(z)
        return _ml_two_scalar_cached(alpha, z.dtype.str, z.tobytes())

    key, z = _ml_two_array_key(z, alpha)
    cached = _ML_TWO_ARRAY_CACHE.get(key)
    if cached is not None:
        _ML_TWO_ARRAY_CACHE.move_to_end(key)
        return cached.copy()

    out = np.asarray(mittag_leffler(z, alpha, alpha))
    cached = np.array(out, copy=True)
    cached.setflags(write=False)
    _ML_TWO_ARRAY_CACHE[key] = cached
    if len(_ML_TWO_ARRAY_CACHE) > ML_TWO_ARRAY_CACHE_SIZE:
        _ML_TWO_ARRAY_CACHE.popitem(last=False)
    return out
