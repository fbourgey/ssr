from collections.abc import Callable
from math import gamma

from numba import njit, prange
import numpy as np
from scipy.linalg import blas
from scipy.special import hyp2f1

from model import (
    ForwardVarianceModel,
    MonteCarloConfig,
    SsrMonteCarloConfig,
    require_params,
    resolve_mc_config,
    resolve_ssr_mc_config,
    validate_interval,
    validate_positive,
    validate_positive_n_quad,
)
from utils import cholesky_from_svd, gauss_legendre, jacobi_quadrature


@njit(cache=True, parallel=True)
def _rough_bergomi_integrals_from_normal_jit(
    normal,
    xi0_t,
    drift_t,
    ssr_shift_factor_t,
    dt,
    eta,
    eval_ssr,
):
    n_disc = xi0_t.shape[0] - 1
    n_paths = normal.shape[1]
    int_v_dt = np.empty(n_paths)
    int_sqrt_v_dw = np.empty(n_paths)
    int_v_dt_shifted = np.empty(n_paths)
    int_sqrt_v_dw_shifted = np.empty(n_paths)

    for j in prange(n_paths):
        sum_v_dt = 0.0
        sum_sqrt_v_dw = 0.0
        sum_v_dt_shifted = 0.0
        sum_sqrt_v_dw_shifted = 0.0

        for k in range(n_disc):
            y_prev = 0.0
            if k > 0:
                y_prev = normal[k - 1, j]
            y_next = normal[k, j]

            exp_prev = np.exp(eta * y_prev + drift_t[k])
            exp_next = np.exp(eta * y_next + drift_t[k + 1])
            v_prev = xi0_t[k] * exp_prev
            v_next = xi0_t[k + 1] * exp_next
            sqrt_v_prev = np.sqrt(v_prev)

            dw = normal[n_disc + k, j]
            if k > 0:
                dw -= normal[n_disc + k - 1, j]

            sum_v_dt += v_prev + v_next
            sum_sqrt_v_dw += sqrt_v_prev * dw

            if eval_ssr:
                shift_prev = ssr_shift_factor_t[k]
                shift_next = ssr_shift_factor_t[k + 1]
                sum_v_dt_shifted += shift_prev * v_prev + shift_next * v_next
                sum_sqrt_v_dw_shifted += np.sqrt(shift_prev) * sqrt_v_prev * dw

        int_v_dt[j] = 0.5 * dt * sum_v_dt
        int_sqrt_v_dw[j] = sum_sqrt_v_dw
        int_v_dt_shifted[j] = 0.5 * dt * sum_v_dt_shifted
        int_sqrt_v_dw_shifted[j] = sum_sqrt_v_dw_shifted

    return int_v_dt, int_sqrt_v_dw, int_v_dt_shifted, int_sqrt_v_dw_shifted


class RoughBergomiModel(ForwardVarianceModel):
    """
    Rough Bergomi model.
    """

    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray],
        s0: float = 1.0,
    ) -> None:
        """
        Initialize Rough Bergomi model.

        Parameters
        ----------
        xi0 : callable
            Initial forward variance curve function.
        params : dict
            Dictionary containing model parameters:
            - eta: volatility of volatility
            - H: Hurst parameter
            - rho: correlation between Brownian motions
        s0 : float, optional
            Initial stock price, by default 1.0
        """
        self.eta, self.H, self.rho = require_params(params, ("eta", "H", "rho"))

        super().__init__(params=params, xi0=xi0, s0=s0)
        self._check_params()
        self._cholesky_cache = {}
        self._unit_cholesky_cache = {}
        self.A_H = (
            np.sqrt(2 * self.H)
            / (4 * (self.H + 0.5) * (self.H + 1.5))
            * (
                ((self.H + 1.5) ** 2 / (self.H + 1))
                * (1 + gamma(self.H + 1.5) ** 2 / gamma(2 * self.H + 2))
                - 3
            )
        )  # constant for SSR expansion

    def _check_params(self):
        """Validate Rough Bergomi parameters."""
        validate_positive("eta", self.eta)
        validate_interval("rho", self.rho, -1, 1)
        validate_interval("H", self.H, 0, 1, closed=False)

    def lbd_xi(self, u, t):
        return (
            self.eta * np.sqrt(2.0 * self.H) * (u - t) ** (self.H - 0.5) * self.xi0(u)
        )

    def cov_levy_fbm(self, u, v):
        r"""
        Compute the covariance matrix of Levy's fractional Brownian motion.

        It corresponds to:

        Cov(W_u^H, W_v^H) = \int_0^{min(u,v)} (u-s)^{H-1/2} (v-s)^{H-1/2} ds

        where:

        W_u^H = \int_0^u (u-s)^{H-1/2} dW_s

        Parameters
        ----------
        u : np.ndarray or float
            First set of time points.
        v : np.ndarray or float
            Second set of time points.

        Returns
        -------
        np.ndarray
            Covariance matrix evaluated at (u, v).
        """
        u_max_v = np.maximum(u, v)
        u_min_v = np.minimum(u, v)
        cov = (
            u_min_v ** (self.H + 0.5)
            * u_max_v ** (self.H - 0.5)
            * hyp2f1(1.0, 0.5 - self.H, 1.5 + self.H, u_min_v / u_max_v)
        )
        return cov / (self.H + 0.5)

    def cholesky_cov_matrix(
        self, tab_t, return_cov: bool = False, conditioning: bool = False
    ):
        r"""
        Compute the lower-triangular Cholesky factor
        of the covariance matrix of the Gaussian vector (Y_{t_i}, W_{t_i})
        for 1 <= i <= n, where t_i are the timesteps in tab_t.

        Here, W is a standard Brownian motion and
        Y_t = \sqrt{2H} \int_0^t (t-s)^{H-1/2} dW_s.

        Parameters
        ----------
        tab_t : np.ndarray
            Array of time grid points (shape: n_steps + 1,).
        return_cov : bool, optional
            If True, return the full covariance matrix instead of its Cholesky factor.
            Default is False.
        conditioning : bool, optional
            If True, compute the conditional covariance (see Bergomi's book, Chapter
            8, Appendix A).
            Default is False.

        Returns
        -------
        np.ndarray
            Lower-triangular Cholesky factor of the covariance matrix, or the covariance
            matrix itself if `return_cov` is True.
        """
        n_disc = tab_t.shape[0] - 1
        # repeat tab_t[1:] n_disc times as columns (shape: n_disc x n_disc)
        u = np.tile(tab_t[1:], (n_disc, 1)).T
        cov_y = 2.0 * self.H * self.cov_levy_fbm(u, u.T)
        cov_w = np.minimum(u, u.T)
        cov_yw = u ** (self.H + 0.5) - (u - cov_w) ** (self.H + 0.5)
        cov_yw *= np.sqrt(2.0 * self.H) / (self.H + 0.5)
        if not conditioning:
            cov_yw *= self.rho
        cov = np.block(
            [
                [cov_y, cov_yw],
                [cov_yw.T, cov_w],
            ]
        )
        if return_cov:
            return cov
        try:
            chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            chol = cholesky_from_svd(cov)
        except Exception as e:
            print(f"Error in Cholesky decomposition: {e}")
            raise

        return chol

    def _cached_cholesky_cov_matrix(self, tab_t, conditioning: bool = False):
        chol, scales = self._cached_cholesky_transform(tab_t, conditioning)
        if scales is None:
            return chol

        scaled_chol = scales[:, np.newaxis] * chol
        scaled_chol.setflags(write=False)
        return scaled_chol

    def _cached_cholesky_transform(self, tab_t, conditioning: bool = False):
        tab_t = np.ascontiguousarray(tab_t, dtype=float)
        unit_chol = self._cached_unit_cholesky_cov_matrix(tab_t, conditioning)
        if unit_chol is not None:
            T = float(tab_t[-1])
            n_disc = tab_t.shape[0] - 1
            scales = np.empty(2 * n_disc)
            scales[:n_disc] = T**self.H
            scales[n_disc:] = T**0.5
            return unit_chol, scales

        key = (conditioning, tab_t.dtype.str, tab_t.shape, tab_t.tobytes())
        chol = self._cholesky_cache.get(key)
        if chol is None:
            chol = self.cholesky_cov_matrix(tab_t, conditioning=conditioning)
            chol.setflags(write=False)
            self._cholesky_cache[key] = chol
        return chol, None

    def _cached_unit_cholesky_cov_matrix(self, tab_t, conditioning: bool):
        T = float(tab_t[-1])
        n_disc = tab_t.shape[0] - 1
        if T <= 0.0 or n_disc <= 0:
            return None

        unit_grid = np.linspace(0.0, 1.0, n_disc + 1)
        if not np.allclose(tab_t / T, unit_grid, rtol=1e-12, atol=1e-14):
            return None

        key = (conditioning, n_disc)
        chol = self._unit_cholesky_cache.get(key)
        if chol is None:
            chol = self.cholesky_cov_matrix(unit_grid, conditioning=conditioning)
            chol.setflags(write=False)
            self._unit_cholesky_cache[key] = chol
        return chol

    def simulate_mc(
        self,
        tab_t: np.ndarray,
        n_mc: int,
        n_loop: int = 1,
        seed: int | None = None,
        conditioning: bool = False,
        eps_ssr: float = 0.0,
    ) -> dict:
        rng = np.random.default_rng(seed)

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")

        eval_ssr = eps_ssr != 0.0
        h_half = self.H == 0.5
        n_disc = tab_t.shape[0] - 1
        dt = tab_t[1] - tab_t[0]

        int_v_dt = np.zeros(n_mc)
        int_sqrt_v_dw = np.zeros(n_mc)

        # Precompute deterministic quantities once
        tab_t_col = tab_t[:, np.newaxis]
        xi0_t = self.xi0(tab_t_col)
        xi0_t_flat = np.asarray(self.xi0(tab_t), dtype=float)
        xi0_prev = xi0_t[:-1, :]
        xi0_next = xi0_t[1:, :]
        drift_t = -0.5 * self.eta**2 * tab_t_col ** (2.0 * self.H)
        drift_t_flat = -0.5 * self.eta**2 * tab_t ** (2.0 * self.H)
        ssr_shift_factor_t_flat = np.ones(n_disc + 1)

        if eval_ssr:
            int_v_dt_shifted = np.zeros(n_mc)
            int_sqrt_v_dw_shifted = np.zeros(n_mc)
            if h_half:
                ssr_shift_factor = np.exp(
                    eps_ssr * self.eta * self.rho / self.xi0_0**0.5
                )
                ssr_sqrt_shift_factor = ssr_shift_factor**0.5
            else:
                c = (
                    eps_ssr
                    * self.eta
                    * self.rho
                    * np.sqrt(2.0 * self.H)
                    / self.xi0_0**0.5
                )
                ssr_shift_factor_t = np.ones_like(tab_t_col)
                mask = tab_t_col != 0.0
                ssr_shift_factor_t[mask] += c * tab_t_col[mask] ** (self.H - 0.5)
                ssr_shift_factor_t_flat = ssr_shift_factor_t.ravel()
                ssr_shift_factor_prev = ssr_shift_factor_t[:-1, :]
                ssr_shift_factor_next = ssr_shift_factor_t[1:, :]
                ssr_sqrt_shift_factor_prev = np.sqrt(ssr_shift_factor_prev)

        # Precompute Cholesky once (expensive)
        if not h_half:
            chol, chol_scales = self._cached_cholesky_transform(
                tab_t, conditioning=conditioning
            )

        sqrt_dt = np.sqrt(dt)
        rho_perp = np.sqrt(1.0 - self.rho**2)

        for i in range(n_loop):
            sl = slice(i * n_mc_loop, (i + 1) * n_mc_loop)

            if h_half:
                dy = rng.normal(0.0, sqrt_dt, (n_disc, n_mc_loop))
                y = np.empty((n_disc + 1, n_mc_loop))
                y[0, :] = 0.0
                np.cumsum(dy, axis=0, out=y[1:, :])

                if conditioning:
                    dw = dy
                elif rho_perp == 0.0:
                    dw = self.rho * dy
                else:
                    dz = rng.normal(0.0, sqrt_dt, (n_disc, n_mc_loop))
                    dw = self.rho * dy + rho_perp * dz
            else:
                normal = rng.normal(0, 1, (2 * n_disc, n_mc_loop))
                normal = blas.dtrmm(
                    1.0,
                    chol,
                    normal,
                    side=0,
                    lower=1,
                    trans_a=0,
                    diag=0,
                )
                if chol_scales is not None:
                    normal *= chol_scales[:, np.newaxis]
                (
                    int_v_dt[sl],
                    int_sqrt_v_dw[sl],
                    int_v_dt_shifted_loop,
                    int_sqrt_v_dw_shifted_loop,
                ) = _rough_bergomi_integrals_from_normal_jit(
                    normal,
                    xi0_t_flat,
                    drift_t_flat,
                    ssr_shift_factor_t_flat,
                    dt,
                    self.eta,
                    eval_ssr,
                )
                if eval_ssr:
                    int_v_dt_shifted[sl] = int_v_dt_shifted_loop
                    int_sqrt_v_dw_shifted[sl] = int_sqrt_v_dw_shifted_loop
                continue

            y *= self.eta
            y += drift_t
            np.exp(y, out=y)
            exp_term = y

            exp_prev = exp_term[:-1, :]
            exp_next = exp_term[1:, :]
            v_prev = xi0_prev * exp_prev
            v_next = xi0_next * exp_next
            sqrt_v_prev = np.sqrt(v_prev)
            int_sqrt_v_dw[sl] = np.sum(sqrt_v_prev * dw, axis=0)
            int_v_dt[sl] = 0.5 * dt * np.sum(v_prev + v_next, axis=0)

            if eval_ssr:
                if h_half:
                    int_sqrt_v_dw_shifted[sl] = (
                        ssr_sqrt_shift_factor * int_sqrt_v_dw[sl]
                    )
                    int_v_dt_shifted[sl] = ssr_shift_factor * int_v_dt[sl]
                else:
                    int_sqrt_v_dw_shifted[sl] = np.sum(
                        ssr_sqrt_shift_factor_prev * sqrt_v_prev * dw, axis=0
                    )
                    int_v_dt_shifted[sl] = (
                        0.5
                        * dt
                        * np.sum(
                            ssr_shift_factor_prev * v_prev
                            + ssr_shift_factor_next * v_next,
                            axis=0,
                        )
                    )

        out = {
            "int_v_dt": int_v_dt,
            "int_sqrt_v_dw": int_sqrt_v_dw,
        }

        if eval_ssr:
            out["int_v_dt_shifted"] = int_v_dt_shifted
            out["int_sqrt_v_dw_shifted"] = int_sqrt_v_dw_shifted

        return out

    def impvol(
        self,
        k,
        T: float,
        config: MonteCarloConfig | None = None,
        *,
        return_skew: bool = False,
        **mc_kwargs,
    ):
        config = resolve_mc_config(config, mc_kwargs)
        return self.impvol_mc(
            k=k,
            T=T,
            n_mc=config.n_mc,
            n_disc=config.n_disc,
            n_loop=config.n_loop,
            seed=config.seed,
            conditioning=config.conditioning,
            return_skew=return_skew,
            rho_cond=self.rho if config.conditioning else None,
        )

    def ssr_all(
        self,
        T,
        config: SsrMonteCarloConfig | None = None,
        **mc_kwargs,
    ):
        config = resolve_ssr_mc_config(config, mc_kwargs)
        return self.ssr_mc_all(
            T=T,
            n_mc=config.n_mc,
            n_disc=config.n_disc,
            eps_ssr=config.eps_ssr,
            n_loop=config.n_loop,
            seed=config.seed,
            conditioning=config.conditioning,
            n_quad=config.n_quad,
            rho_cond=self.rho if config.conditioning else None,
            n_batch=config.n_batch,
        )

    def xi0_shifted(self, u, eps):
        """Shifted initial forward variance curve for finite difference SSR."""
        u = np.asarray(u, dtype=float)
        if self.H == 0.5:
            out = self.xi0(u) * np.exp(eps * self.eta * self.rho / self.xi0_0**0.5)
        else:
            out = np.full(u.shape, self.xi0_0, dtype=float)
            mask = u != 0.0
            if np.any(mask):
                c = eps * self.eta * self.rho * np.sqrt(2.0 * self.H) / self.xi0_0**0.5
                um = u[mask]
                out[mask] = self.xi0(um) * (1.0 + c * um ** (self.H - 0.5))

        return out

    def mu_bg(self, t, u):
        return (
            self.xi0(t) ** 0.5
            * self.eta
            * self.rho
            * np.sqrt(2.0 * self.H)
            * self.xi0(u)
            * (u - t) ** (self.H - 0.5)
        )

    def nu_bg(self, t, u, v) -> float | np.ndarray:
        return (
            2.0
            * self.H
            * self.eta**2
            * self.xi0(u)
            * self.xi0(v)
            * ((u - t) * (v - t)) ** (self.H - 0.5)
        )

    def _beta_1_gauss_quad(self, T, n_quad: int):
        """First-order beta coefficient using Gauss quadrature."""
        validate_positive_n_quad(n_quad)

        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis]
        t_grid, w_t = jacobi_quadrature(n_quad, 0, self.H - 0.5)
        t_grid = t_grid[np.newaxis, :]
        w_t = w_t[np.newaxis, :]

        integral = np.sum(w_t * self.xi0(0.5 * T_grid * (1 + t_grid)), axis=1)
        integral *= (
            np.sqrt(2 * self.H)
            * self.rho
            * self.eta
            * T ** (self.H - 0.5)
            / (2 ** (self.H + 0.5) * self.xi0_0**0.5)
        )
        return integral / (2 * self.sig_varswap(T))

    def _c_x_xi_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)

        T = np.atleast_1d(T)
        s_grid, w_s = gauss_legendre(0.0, 1.0, n_quad)
        u_grid, w_u = jacobi_quadrature(n_quad, 0.0, self.H - 0.5)
        s_grid, u_grid = np.meshgrid(s_grid, u_grid, indexing="ij")
        w_s = w_s[np.newaxis, :, np.newaxis]
        w_u = w_u[np.newaxis, np.newaxis, :]
        T_grid = T[:, np.newaxis, np.newaxis]

        out = np.sum(
            w_s
            * w_u
            * self.xi0(T_grid * s_grid) ** 0.5
            * (1 - s_grid) ** (self.H + 0.5)
            * self.xi0(0.5 * T_grid * ((1 - s_grid) * u_grid + 1 + s_grid)),
            axis=(1, 2),
        )
        out *= (
            self.eta
            * self.rho
            * np.sqrt(2.0 * self.H)
            * T ** (self.H + 1.5)
            / 2 ** (self.H + 0.5)
        )
        return out

    def _c_mu_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)

        # common terms for the two integrals
        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis, np.newaxis, np.newaxis]

        # first integral
        t1_grid, w_t1 = gauss_legendre(0.0, 1.0, n_quad)
        y1_grid, w_y1 = jacobi_quadrature(n_quad, self.H - 0.5, 0.0)
        z1_grid, w_z1 = jacobi_quadrature(n_quad, 0.0, 2 * self.H)
        t1_grid, z1_grid, y1_grid = np.meshgrid(
            t1_grid, z1_grid, y1_grid, indexing="ij"
        )
        w_t1 = w_t1[np.newaxis, :, np.newaxis, np.newaxis]
        w_z1 = w_z1[np.newaxis, np.newaxis, :, np.newaxis]
        w_y1 = w_y1[np.newaxis, np.newaxis, np.newaxis, :]

        # second integral
        t2_grid, w_t2 = gauss_legendre(0.0, 1.0, n_quad)
        z2_grid, w_z2 = jacobi_quadrature(n_quad, self.H + 0.5, self.H - 0.5)
        y2_grid, w_y2 = jacobi_quadrature(n_quad, 0.0, self.H - 0.5)
        t2_grid, z2_grid, y2_grid = np.meshgrid(
            t2_grid, z2_grid, y2_grid, indexing="ij"
        )
        w_t2 = w_t2[np.newaxis, :, np.newaxis, np.newaxis]
        w_z2 = w_z2[np.newaxis, np.newaxis, :, np.newaxis]
        w_y2 = w_y2[np.newaxis, np.newaxis, np.newaxis, :]

        out1 = np.sum(
            w_t1
            * w_z1
            * w_y1
            * self.xi0(T_grid * t1_grid) ** 0.5
            * (1 - t1_grid) ** (2 * self.H + 1)
            * self.xi0(0.5 * T_grid * ((1 - t1_grid) * z1_grid + 1 + t1_grid))
            * self.xi0(
                0.25
                * T_grid
                * (
                    1
                    + y1_grid * (1 + z1_grid)
                    + z1_grid
                    + t1_grid * (3 - y1_grid * (1 + z1_grid) - z1_grid)
                )
            )
            ** 0.5,
            axis=(1, 2, 3),
        )
        out2 = 0.5 * np.sum(
            w_t2
            * w_z2
            * w_y2
            * self.xi0(T_grid * t1_grid) ** 0.5
            * (1 - t1_grid) ** (2 * self.H + 1)
            * self.xi0(0.5 * T_grid * ((1 - t2_grid) * z2_grid + 1 + t2_grid)) ** 0.5
            * self.xi0(
                0.25
                * T_grid
                * (3 + y2_grid + (1 - y2_grid) * (t2_grid * (1 - z2_grid) + z2_grid))
            ),
            axis=(1, 2, 3),
        )
        out = (
            2
            * self.H
            * self.rho**2
            * self.eta**2
            * T ** (2 + 2 * self.H)
            / 2 ** (3 * self.H + 1.5)
        ) * (out1 + out2)

        return out

    def _dxi_c_x_xi_beta_2_gauss_quad(self, T, n_quad: int) -> float | np.ndarray:
        """
        Fréchet derivative of C^{x,xi} with respect to xi0 in the direction dxi.
        This is used in the calculation of beta_2_bg.
        """
        validate_positive_n_quad(n_quad)

        # common terms for the two integrals
        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis, np.newaxis]

        # first integral
        t1_grid, w_t1 = gauss_legendre(0.0, 1.0, n_quad)
        y1_grid, w_y1 = jacobi_quadrature(n_quad, self.H - 0.5, self.H - 0.5)
        t1_grid, y1_grid = np.meshgrid(t1_grid, y1_grid, indexing="ij")
        t1_grid = t1_grid[np.newaxis, :, :]
        y1_grid = y1_grid[np.newaxis, :, :]
        w_t1 = w_t1[np.newaxis, :, np.newaxis]
        w_y1 = w_y1[np.newaxis, np.newaxis, :]

        # second integral
        t2_grid, w_t2 = gauss_legendre(0.0, 1.0, n_quad)
        y2_grid, w_y2 = jacobi_quadrature(n_quad, self.H - 0.5, 0.0)
        t2_grid, y2_grid = np.meshgrid(t2_grid, y2_grid, indexing="ij")
        t2_grid = t2_grid[np.newaxis, :, :]
        y2_grid = y2_grid[np.newaxis, :, :]
        w_t2 = w_t2[np.newaxis, :, np.newaxis]
        w_y2 = w_y2[np.newaxis, np.newaxis, :]

        out1 = np.sum(
            w_t1
            * w_y1
            * self.xi0(T_grid * t1_grid)
            * t1_grid ** (2 * self.H)
            * self.xi0(0.5 * T_grid * t1_grid * (1 + y1_grid)) ** 0.5
            / 2 ** (2 * self.H + 1),
            axis=(1, 2),
        )
        out2 = np.sum(
            w_t2
            * w_y2
            * self.xi0(T_grid * t2_grid)
            * t2_grid ** (2 * self.H)
            * self.xi0(0.5 * T_grid * t2_grid * (1 + y2_grid)) ** 0.5
            / 2 ** (self.H + 0.5),
            axis=(1, 2),
        )
        out = (
            T ** (2 * self.H + 1)
            * self.eta**2
            * self.rho**2
            * 2
            * self.H
            / self.xi0_0**0.5
        ) * (out1 + out2)

        return out

    ####################################################################################
    # flat forward variance curve and kappa = 0 case
    ####################################################################################

    def _beta_1_bg_flat(self, T):
        return (
            self.eta
            * self.rho
            * np.sqrt(2.0 * self.H)
            * T ** (self.H - 0.5)
            / (2.0 * self.H + 1.0)
        )

    def _beta_2_bg_flat(self, T):
        return (
            self._beta_1_bg_flat(T)
            * self.eta
            * self.rho
            * np.sqrt(2 * self.H)
            / (4 * (self.H + 1.5))
            * np.sqrt(self.xi0_0)
            * (
                1
                + 0.5
                * (self.H + 1.5)
                * gamma(self.H + 0.5) ** 2
                / gamma(2 * self.H + 1)
            )
            * T ** (self.H + 0.5)
        )

    def _atm_skew_1_bg_flat(self, T):
        return self._beta_1_bg_flat(T) / (self.H + 1.5)

    def _atm_skew_2_bg_flat(self, T):
        return (
            self._atm_skew_1_bg_flat(T)
            * self.eta
            * self.rho
            * self.xi0_0**0.5
            * self.A_H
            * T ** (self.H + 0.5)
        )

    def _ssr_0_bg_flat(self):
        return self.H + 1.5

    def _ssr_1_vs_bg_flat(self, T):
        return (
            -(self.H + 1.5)
            * self.eta
            * self.rho
            * self.xi0_0**0.5
            * self.A_H
            * T ** (self.H + 0.5)
        )

    def _ssr_1_bg_flat(self, T):
        return self._ssr_1_vs_bg_flat(T) + self.eta * self.rho * np.sqrt(
            2 * self.H
        ) * np.sqrt(self.xi0_0) / 4 * (
            1 + 0.5 * (self.H + 1.5) * gamma(self.H + 0.5) ** 2 / gamma(2 * self.H + 1)
        ) * T ** (self.H + 0.5)


def get_params_rough_bergomi(id: int):
    """Get Rough Bergomi parameters for a given id."""
    if id not in [1]:
        raise ValueError("Invalid id. Please choose a valid id.")
    if id == 1:
        # Bayer, Friz, Gatheral (2016), Pricing under rough volatility,
        # Quantitative Finance
        params = {"eta": 1.9, "H": 0.05, "rho": -0.9}
        xi0 = lambda t: 0.235**2 * np.ones_like(t)
    return params, xi0
