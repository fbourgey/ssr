from collections.abc import Callable

from numba import float64, njit, prange, vectorize
import numpy as np

from .model import (
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
from .utils import gauss_legendre
from scipy.integrate import quad


@njit(cache=True, parallel=True)
def _two_bergomi_integrals_from_normal_jit(
    normal,
    xi0_t,
    variance_drift_t,
    shift_factor,
    dt,
    l11,
    l21,
    l22,
    l31,
    l32,
    l33,
    decay1,
    decay2,
    x1_0,
    x2_0,
    w,
    alpha,
    weight1,
    weight2,
    eval_ssr,
):
    n_disc = normal.shape[1]
    n_paths = normal.shape[2]
    int_v_dt = np.empty(n_paths)
    int_sqrt_v_dw = np.empty(n_paths)
    int_v_dt_shifted = np.empty(n_paths)
    int_sqrt_v_dw_shifted = np.empty(n_paths)

    for j in prange(n_paths):
        x1_prev = x1_0
        x2_prev = x2_0
        v_prev = xi0_t[0] * np.exp(
            w * alpha * (weight1 * x1_prev + weight2 * x2_prev) - variance_drift_t[0]
        )

        sum_v_dt = 0.0
        sum_sqrt_v_dw = 0.0
        sum_v_dt_shifted = 0.0
        sum_sqrt_v_dw_shifted = 0.0

        for i in range(n_disc):
            z1 = normal[0, i, j]
            z2 = normal[1, i, j]
            z3 = normal[2, i, j]

            dx1 = l11 * z1
            dx2 = l21 * z1 + l22 * z2
            dw = l31 * z1 + l32 * z2 + l33 * z3

            x1_next = decay1 * x1_prev + dx1
            x2_next = decay2 * x2_prev + dx2
            v_next = xi0_t[i + 1] * np.exp(
                w * alpha * (weight1 * x1_next + weight2 * x2_next)
                - variance_drift_t[i + 1]
            )

            sum_sqrt_v_dw += np.sqrt(v_prev) * dw
            sum_v_dt += v_prev + v_next

            if eval_ssr:
                shift_prev = shift_factor[i]
                shift_next = shift_factor[i + 1]
                sum_sqrt_v_dw_shifted += np.sqrt(shift_prev * v_prev) * dw
                sum_v_dt_shifted += shift_prev * v_prev + shift_next * v_next

            x1_prev = x1_next
            x2_prev = x2_next
            v_prev = v_next

        int_v_dt[j] = 0.5 * dt * sum_v_dt
        int_sqrt_v_dw[j] = sum_sqrt_v_dw
        int_v_dt_shifted[j] = 0.5 * dt * sum_v_dt_shifted
        int_sqrt_v_dw_shifted[j] = sum_sqrt_v_dw_shifted

    return int_v_dt, int_sqrt_v_dw, int_v_dt_shifted, int_sqrt_v_dw_shifted


class TwoFactorBergomiModel(ForwardVarianceModel):
    """
    Two-factor Bergomi model.
    """

    def __init__(
        self,
        params: dict,
        xi0: Callable[[np.ndarray], np.ndarray],
        s0: float = 1.0,
    ) -> None:
        """
        Initialize two-factor Bergomi model.

        Parameters
        ----------
        xi0 : callable
            Initial forward variance curve function.
        params : dict
            Dictionary containing model parameters:
            - k1: speed of mean reversion for first OU factor
            - k2: speed of mean reversion for second OU factor
            - theta: mixing parameter between the two factors (in [0, 1])
            - w: volatility of volatility
            - rho12: correlation between the two factors
            - rhoS1: correlation between the first factor and the stock
            - rhoS2: correlation between the second factor and the stock
        s0 : float, optional
            Initial stock price, by default 1.0
        """
        (
            self.k1,
            self.k2,
            self.theta,
            self.w,
            self.rho12,
            self.rhoS1,
            self.rhoS2,
        ) = require_params(
            params, ("k1", "k2", "theta", "w", "rho12", "rhoS1", "rhoS2")
        )

        super().__init__(params=params, xi0=xi0, s0=s0)
        self._check_params()
        self.alpha = (
            (1.0 - self.theta) ** 2
            + 2.0 * self.rho12 * (1.0 - self.theta) * self.theta
            + self.theta**2
        ) ** (-0.5)
        self.chi = (self.rho12 - self.rhoS1 * self.rhoS2) / np.sqrt(
            (1.0 - self.rhoS1**2) * (1.0 - self.rhoS2**2)
        )
        if self.chi < -1 or self.chi > 1:
            raise ValueError(
                "Invalid correlation parameters. Covariance matrix is not positive "
                "semidefinite."
            )
        # initial values of the two OU factors
        self.x1_0 = float(params.get("x1_0", 0.0))
        self.x2_0 = float(params.get("x2_0", 0.0))

        # Correlation parameter for the conditioning case
        # Bergomi (2016) - Stochastic volatility modeling, Equation (8.62b)
        self.rho_cond = np.sqrt(
            (self.rhoS1**2 + self.rhoS2**2 - 2.0 * self.rhoS1 * self.rhoS2 * self.rho12)
            / (1.0 - self.rho12**2)
        )
        if self.rho_cond < -1 or self.rho_cond > 1:
            raise ValueError("Invalid correlation parameter for conditioning.")

    def _check_params(self):
        """Validate two-factor Bergomi parameters."""
        validate_positive("k1", self.k1)
        validate_positive("k2", self.k2)
        validate_positive("w", self.w)
        validate_interval("theta", self.theta, 0, 1)
        validate_interval("rho12", self.rho12, -1, 1, closed=False)
        validate_interval("rhoS1", self.rhoS1, -1, 1, closed=False)
        validate_interval("rhoS2", self.rhoS2, -1, 1, closed=False)

    def lbd_xi(self, u, t) -> float | np.ndarray:
        exp1 = (1 - self.theta) * np.exp(-self.k1 * (u - t))
        exp2 = self.theta * np.exp(-self.k2 * (u - t))
        return self.w * self.alpha * self.xi0(u) * np.array([exp1, exp2])

    def _var(self, t, u):
        """
        Compute the variance of the Gaussian process x_t_u defined as
            alpha * (
                (1 - theta) * exp(-k1 * (u - t)) * X1 + theta * exp(-k2 * (u - t)) * X2
            )
        where X1, X2 are the two OU factors.

        See Bergomi (2016) - Stochastic volatility modeling, Equation (7.35).
        """
        exp1 = t * _func_I(2.0 * self.k1 * t)
        exp1 = (1.0 - self.theta) ** 2 * np.exp(-2.0 * self.k1 * (u - t)) * exp1

        exp2 = t * _func_I(2.0 * self.k2 * t)
        exp2 = self.theta**2 * np.exp(-2.0 * self.k2 * (u - t)) * exp2

        exp12 = t * _func_I((self.k1 + self.k2) * t)
        exp12 = (
            2.0
            * self.theta
            * (1.0 - self.theta)
            * self.rho12
            * np.exp(-(self.k1 + self.k2) * (u - t))
            * exp12
        )
        return self.alpha**2 * (exp1 + exp2 + exp12)

    def _f_xi(self, t, u, x1, x2):
        """
        Helper function to compute the forward variance at time t for maturity u
        given the two OU factors X_t = (x1, x2) where
        xi_t(u) = xi_0(u) * f^u(t, x1, x2).

        See Bergomi (2016) - Stochastic volatility modeling, Equation (7.34).
        """
        exp1 = np.exp(-self.k1 * (u - t))
        exp2 = np.exp(-self.k2 * (u - t))
        x_t_u = self.alpha * ((1 - self.theta) * exp1 * x1 + self.theta * exp2 * x2)
        return np.exp(self.w * x_t_u - 0.5 * self.w**2 * self._var(t, u))

    def simulate_mc(
        self,
        tab_t: np.ndarray,
        n_mc: int,
        n_loop: int = 1,
        seed=None,
        conditioning: bool = False,
        eps_ssr: float = 0.0,
    ) -> dict:

        rng = np.random.default_rng(seed)

        n_mc_loop, remainder = divmod(n_mc, n_loop)
        if remainder != 0:
            raise ValueError("n_mc must be divisible by n_loop")

        eval_ssr = eps_ssr != 0.0
        n_disc = tab_t.shape[0] - 1
        dt = tab_t[1] - tab_t[0]

        int_v_dt = np.zeros(n_mc)
        int_sqrt_v_dw = np.zeros(n_mc)

        # Precompute deterministic quantities once
        xi0_t = np.asarray(self.xi0(tab_t), dtype=float)
        shift_factor = np.ones(n_disc + 1)

        if eval_ssr:
            int_v_dt_shifted = np.zeros(n_mc)
            int_sqrt_v_dw_shifted = np.zeros(n_mc)
            x1_0_shifted = self.x1_0 + eps_ssr * self.rhoS1 / self.xi0_0**0.5
            x2_0_shifted = self.x2_0 + eps_ssr * self.rhoS2 / self.xi0_0**0.5

        # Standard deviations of the increments of the two OU factors dx1, dx2 and the
        # Brownian motion dW driving the spot
        std_dx1 = (dt * _func_I(2.0 * self.k1 * dt)) ** 0.5
        std_dx2 = (dt * _func_I(2.0 * self.k2 * dt)) ** 0.5
        std_dW = dt**0.5
        # Cov(dx1, dx2)
        cov_dx1_dx2 = self.rho12 * dt * _func_I((self.k1 + self.k2) * dt)
        # Cov(dx1, dW)
        cov_dx1_dW = dt * _func_I(self.k1 * dt)
        if conditioning:
            cov_dx1_dW *= (self.rhoS1 - self.rho12 * self.rhoS2) + self.rho12 * (
                self.rhoS2 - self.rho12 * self.rhoS1
            )
            cov_dx1_dW /= self.rho_cond * (1 - self.rho12**2)
        else:
            cov_dx1_dW *= self.rhoS1
        # Cov(dx2, dW)
        cov_dx2_dW = dt * _func_I(self.k2 * dt)
        if conditioning:
            cov_dx2_dW *= (self.rhoS2 - self.rho12 * self.rhoS1) + self.rho12 * (
                self.rhoS1 - self.rho12 * self.rhoS2
            )
            cov_dx2_dW /= self.rho_cond * (1 - self.rho12**2)
        else:
            cov_dx2_dW *= self.rhoS2

        # chol = np.linalg.cholesky(
        #     np.array(
        #         [
        #             [std_dx1**2, cov_dx1_dx2, cov_dx1_dW],
        #             [cov_dx1_dx2, std_dx2**2, cov_dx2_dW],
        #             [cov_dx1_dW, cov_dx2_dW, std_dW**2],
        #         ]
        #     )
        # )

        l11 = std_dx1

        l21 = cov_dx1_dx2 / l11
        l31 = cov_dx1_dW / l11

        s22 = std_dx2**2 - l21**2
        l22 = np.sqrt(s22)

        l32 = (cov_dx2_dW - l21 * l31) / l22

        s33 = std_dW**2 - l31**2 - l32**2
        l33 = np.sqrt(s33)

        decay1 = np.exp(-self.k1 * dt)
        decay2 = np.exp(-self.k2 * dt)
        variance_drift_t = np.asarray(
            0.5 * self.w**2 * self._var(tab_t, tab_t), dtype=float
        )
        weight1 = 1.0 - self.theta
        weight2 = self.theta

        if eval_ssr:
            n_disc_range = np.arange(n_disc + 1, dtype=float)
            exp_k1 = np.exp(-self.k1 * dt * n_disc_range)
            exp_k2 = np.exp(-self.k2 * dt * n_disc_range)
            shift_factor = np.exp(
                self.w
                * self.alpha
                * (
                    weight1 * (x1_0_shifted - self.x1_0) * exp_k1
                    + weight2 * (x2_0_shifted - self.x2_0) * exp_k2
                )
            )

        for i in range(n_loop):
            sl = slice(i * n_mc_loop, (i + 1) * n_mc_loop)
            normal = rng.normal(0, 1, (3, n_disc, n_mc_loop))

            (
                int_v_dt[sl],
                int_sqrt_v_dw[sl],
                int_v_dt_shifted_loop,
                int_sqrt_v_dw_shifted_loop,
            ) = _two_bergomi_integrals_from_normal_jit(
                normal,
                xi0_t,
                variance_drift_t,
                shift_factor,
                dt,
                l11,
                l21,
                l22,
                l31,
                l32,
                l33,
                decay1,
                decay2,
                self.x1_0,
                self.x2_0,
                self.w,
                self.alpha,
                weight1,
                weight2,
                eval_ssr,
            )

            if eval_ssr:
                int_v_dt_shifted[sl] = int_v_dt_shifted_loop
                int_sqrt_v_dw_shifted[sl] = int_sqrt_v_dw_shifted_loop

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
            rho_cond=self.rho_cond if config.conditioning else None,
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
            rho_cond=self.rho_cond if config.conditioning else None,
            n_batch=config.n_batch,
        )

    def ssr_bg_all(self, T: float | np.ndarray, n_quad: int, **kwargs) -> dict:
        """
        Compute SSR, VS SSR, beta, and ATM skew from the Bergomi-Guyon expansion
        up to second order.
        """
        T = np.atleast_1d(np.asarray(T, dtype=float))

        if self.xi0_flat and not np.isclose(self.k1, self.k2):
            return self._ssr_bg_all_from_expansion_terms(
                beta_1=self._beta_1_bg_flat(T),
                beta_2=self._beta_2_bg_flat(T),
                atm_skew_1=self._atm_skew_1_bg_flat(T),
                atm_skew_2=self._atm_skew_2_bg_flat(T),
            )
        else:
            _sig_vs = self.sig_varswap(T, n_quad=n_quad)
            _c_x_xi_bg = self._c_x_xi_bg_gauss_quad(T, n_quad=n_quad)
            beta_1 = self._beta_1_gauss_quad_from_sig_varswap(
                T, n_quad=n_quad, sig_varswap=_sig_vs
            )
            return self._ssr_bg_all_from_quadrature_terms(
                T=T,
                n_quad=n_quad,
                sig_vs=_sig_vs,
                c_x_xi_bg=_c_x_xi_bg,
                beta_1=beta_1,
            )

    def mu_bg(self, t, u) -> float | np.ndarray:
        """
        Equation (8.50) in 'Bergomi - Stochastic volatility modeling - 2016'.
        """
        return (
            self.xi0(t) ** 0.5
            * self.w
            * self.alpha
            * self.xi0(u)
            * self._factor_kernel(u - t)
        )

    def nu_bg(self, t, u, v) -> float | np.ndarray:
        """
        Equation (8.51) in 'Bergomi - Stochastic volatility modeling - 2016'.
        """
        out = (1 - self.theta) ** 2 * np.exp(-self.k1 * (u + v - 2 * t))
        out += self.theta**2 * np.exp(-self.k2 * (u + v - 2 * t))
        out += (
            self.rho12
            * self.theta
            * (1 - self.theta)
            * (
                np.exp(-self.k1 * (u - t) - self.k2 * (v - t))
                + np.exp(-self.k2 * (u - t) - self.k1 * (v - t))
            )
        )
        return self.w**2 * self.alpha**2 * self.xi0(u) * self.xi0(v) * out

    def _lbd1(self, t, u):
        """
        lbd_1(t,u) function in the Bergomi-Guyon expansion.
        See Bergomi - Guyon (2011), The smile in stochastic volatility models.
        """
        return self.xi0(u) * self.w * self.alpha * self._factor_kernel(u - t)

    def _dlbd1_dxi(self, t, u):
        """
        dlbd_1(t,u) / dxi^u function in the Bergomi-Guyon expansion.
        See Bergomi - Guyon (2011), The smile in stochastic volatility models.
        """
        return self._lbd1(t, u) / self.xi0(u)

    def _factor_kernel(self, tau):
        return (1.0 - self.theta) * self.rhoS1 * np.exp(
            -self.k1 * tau
        ) + self.theta * self.rhoS2 * np.exp(-self.k2 * tau)

    def _beta_1_gauss_quad(self, T, n_quad: int):
        """First-order beta coefficient using Gauss quadrature."""
        validate_positive_n_quad(n_quad)
        T = np.atleast_1d(np.asarray(T, dtype=float))
        if self.xi0_flat:
            return self._beta_1_bg_flat(T)
        return self._beta_1_gauss_quad_from_sig_varswap(
            T, n_quad=n_quad, sig_varswap=self.sig_varswap(T, n_quad)
        )

    def _beta_1_gauss_quad_from_sig_varswap(
        self, T, n_quad: int, sig_varswap
    ) -> float | np.ndarray:
        validate_positive_n_quad(n_quad)
        x_leg, w_leg = gauss_legendre(0, 1, n_quad)
        integral = (
            np.sum(
                w_leg[np.newaxis, :]
                * self.mu_bg(0, T[:, np.newaxis] * x_leg[np.newaxis, :]),
                axis=1,
            )
            / self.xi0_0
        )
        return integral / (2 * sig_varswap)

    def _c_x_xi_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        Gauss-Legendre quadrature implementation of the C^{x,xi} functional for the
        Bergomi-Guyon expansion.
        """
        if self.xi0_flat:
            return self._c_x_xi_bg_flat(np.atleast_1d(np.asarray(T, dtype=float)))
        return self._c_x_xi_bg_gauss_leg_quad(T, n_quad)

    def _c_x_xi_bg_gauss_quad_explicit(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        Gauss-Legendre quadrature implementation of the C^{x,xi} functional for the
        Bergomi-Guyon expansion.
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
            * self.xi0(T_grid * u_grid)
            * self.xi0(T_grid * u_grid * s_grid) ** 0.5
            * self._factor_kernel(T_grid * u_grid * (1 - s_grid)),
            axis=(1, 2),
        )
        return T**2 * self.alpha * self.w * out

    def _c_mu_bg_scipy_quad(self, T: float | np.ndarray) -> float | np.ndarray:

        def _integrand_1(s, u, Ti):
            out1 = quad(lambda t: self._lbd1(u, t), u, Ti)[0] / (2 * self.xi0(u) ** 0.5)
            out2 = quad(lambda r: self.xi0(r) ** 0.5 * self._dlbd1_dxi(r, u), s, u)[0]
            return out1 + out2

        def _integrand_2(s, Ti):
            return quad(
                lambda u: (
                    self.xi0(s) ** 0.5 * self._lbd1(s, u) * _integrand_1(s, u, Ti)
                ),
                s,
                Ti,
            )[0]

        return np.array(
            [
                quad(lambda s, Ti=Ti: _integrand_2(s, Ti), 0, Ti)[0]
                for Ti in np.atleast_1d(T)
            ]
        )

    def _c_mu_bg_gauss_quad(
        self, T: float | np.ndarray, n_quad: int
    ) -> float | np.ndarray:
        """
        C^{mu} functional for the Bergomi-Guyon expansion using Gauss quadrature.
        """
        validate_positive_n_quad(n_quad)
        if self.xi0_flat and not np.isclose(self.k1, self.k2):
            return self._c_mu_bg_flat(np.atleast_1d(np.asarray(T, dtype=float)))

        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis, np.newaxis, np.newaxis]

        knots, weights = gauss_legendre(0.0, 1.0, n_quad)
        s_knots = knots[:, np.newaxis, np.newaxis]
        u_unit = knots[np.newaxis, :, np.newaxis]
        v_unit = knots[np.newaxis, np.newaxis, :]
        s_weights = weights[:, np.newaxis, np.newaxis]
        u_weights = weights[np.newaxis, :, np.newaxis]
        v_weights = weights[np.newaxis, np.newaxis, :]

        u_knots = s_knots + (1.0 - s_knots) * u_unit
        u_weights = (1.0 - s_knots) * u_weights
        v_knots = u_knots + (1.0 - u_knots) * v_unit
        v_weights_1 = (1.0 - u_knots) * v_weights
        mu_s_u = self.mu_bg(T_grid * s_knots, T_grid * u_knots)

        c_mu_1 = 0.5 * np.sum(
            s_weights
            * u_weights
            * v_weights_1
            * self.xi0(T_grid * u_knots) ** (-0.5)
            * mu_s_u
            * self.xi0(T_grid * v_knots[np.newaxis, :, :, :])
            * self._factor_kernel(T_grid * (v_knots - u_knots)),
            axis=(1, 2, 3),
        )

        # C_mu_2
        v_knots = s_knots + (u_knots - s_knots) * v_unit
        v_weights_2 = (u_knots - s_knots) * v_weights
        c_mu_2 = np.sum(
            s_weights
            * u_weights
            * v_weights_2
            * mu_s_u
            * self.xi0(T_grid * v_knots[np.newaxis, :, :, :]) ** 0.5
            * self._factor_kernel(T_grid * (u_knots - v_knots)),
            axis=(1, 2, 3),
        )
        return T**3 * self.w * self.alpha * (c_mu_1 + c_mu_2)

    def _dxi_c_x_xi_beta_2_gauss_quad(self, T, n_quad: int) -> float | np.ndarray:
        """
        Fréchet derivative of C^{x,xi} with respect to xi0 in the direction dxi.
        This is used in the calculation of beta_2_bg.
        """
        validate_positive_n_quad(n_quad)

        # common terms for the two integrals
        T = np.atleast_1d(T)
        T_grid = T[:, np.newaxis, np.newaxis]

        s_grid, w_s = gauss_legendre(0.0, 1.0, n_quad)
        u_grid, w_u = gauss_legendre(0.0, 1.0, n_quad)
        s_grid, u_grid = np.meshgrid(s_grid, u_grid, indexing="ij")
        u_grid = u_grid[np.newaxis, :, :]
        s_grid = s_grid[np.newaxis, :, :]
        w_s = w_s[np.newaxis, :, np.newaxis]
        w_u = w_u[np.newaxis, np.newaxis, :]

        def _integrand(s, u):
            def _foo(x):
                out = (1 - self.theta) * np.exp(
                    -self.k1 * x
                ) * self.rhoS1 + self.theta * np.exp(-self.k2 * x) * self.rhoS2
                return self.w * self.alpha * out

            return (
                self.xi0(s) ** 0.5
                * self.xi0(u)
                * (0.5 * _foo(s) + _foo(u))
                * _foo(u - s)
                / self.xi0_0**0.5
            )

        integral = np.sum(
            w_s * w_u * u_grid * _integrand(T_grid * s_grid * u_grid, T_grid * u_grid),
            axis=(1, 2),
        )
        return T**2 * integral

    ####################################################################################
    # flat forward variance curve and kappa = 0 case
    ####################################################################################

    def _c_x_xi_bg_flat(self, T):
        """
        C^{x,xi} functional for the Bergomi-Guyon expansion with flat
        forward variance curve.  See Section 6.1 in Bergomi and Guyon (2011) -
        The smile in stochastic volatility models.
        """
        return (
            self.alpha
            * self.xi0_0**1.5
            * self.w
            * T**2
            * (
                (1 - self.theta) * self.rhoS1 * _func_J(self.k1 * T)
                + self.theta * self.rhoS2 * _func_J(self.k2 * T)
            )
        )

    def _c_xi_xi_bg_flat(self, T):
        """
        C^{xi,xi} functional for the Bergomi-Guyon expansion with flat forward variance
        curve. See Section 6.1 in Bergomi and Guyon (2011) - The smile in stochastic
        volatility models.
        """
        w1X = (1 - self.theta) * self.rhoS1
        w2X = (1 - self.theta) * (1 - self.rhoS1**2) ** 0.5
        w3X = 0.0

        w1Y = self.theta * self.rhoS2
        w2Y = self.theta * self.chi * (1 - self.rhoS2**2) ** 0.5
        w3Y = self.theta * ((1 - self.chi**2) * (1 - self.rhoS2**2)) ** 0.5

        wX_vec = np.array([w1X, w2X, w3X])
        wY_vec = np.array([w1Y, w2Y, w3Y])

        w0 = np.sum((wX_vec / self.k1 + wY_vec / self.k2) ** 2)
        wX = -2 * np.sum((wX_vec / self.k1) * (wX_vec / self.k1 + wY_vec / self.k2))
        wY = -2 * np.sum((wY_vec / self.k2) * (wX_vec / self.k1 + wY_vec / self.k2))
        wXX = np.sum((wX_vec / self.k1) ** 2)
        wYY = np.sum((wY_vec / self.k2) ** 2)
        wXY = 2 * np.sum((wX_vec / self.k1) * (wY_vec / self.k2))

        out = (
            w0
            + wX * _func_I(self.k1 * T)
            + wY * _func_I(self.k2 * T)
            + wXX * _func_I(2 * self.k1 * T)
            + wYY * _func_I(2 * self.k2 * T)
            + wXY * _func_I((self.k1 + self.k2) * T)
        )
        return self.alpha**2 * self.xi0_0**2 * T * out

    def _c_mu_bg_flat(self, T):
        """
        C^{mu} functional for the Bergomi-Guyon expansion with flat
        forward variance curve.
        See Section 6.1 in Bergomi and Guyon (2011) - The smile in stochastic
        volatility models.
        """

        w1X = (1 - self.theta) * self.rhoS1
        w1Y = self.theta * self.rhoS2

        wX_pp = w1X**2 / (self.k1 * T) + (w1X * w1Y) / (self.k2 * T)
        wY_pp = w1Y**2 / (self.k2 * T) + (w1X * w1Y) / (self.k1 * T)

        wXX_pp = -(w1X**2) / (self.k1 * T)
        wYY_pp = -(w1Y**2) / (self.k2 * T)
        wXY_pp = -(w1X * w1Y) * (1 / (self.k1 * T) + 1 / (self.k2 * T))

        c1 = (
            0.5 * w1X**2 * _func_H(self.k1 * T)
            + 0.5 * w1Y**2 * _func_H(self.k2 * T)
            - w1X
            * w1Y
            * (_func_J(self.k2 * T) - _func_J(self.k1 * T))
            / ((self.k2 - self.k1) * T)
        )
        c2 = (
            wX_pp * _func_J(self.k1 * T)
            + wY_pp * _func_J(self.k2 * T)
            + wXX_pp * _func_J(2 * self.k1 * T)
            + wYY_pp * _func_J(2 * self.k2 * T)
            + wXY_pp * _func_J((self.k1 + self.k2) * T)
        )

        return self.alpha**2 * self.w**2 * self.xi0_0**2 * T**3 * (c1 + c2)

    def _c_mu_bg_flat_alt(self, T):
        """
        Alternative computation for C^{mu} functional for the Bergomi-Guyon expansion
        with flat forward variance curve.
        """
        w1X = (1 - self.theta) * self.rhoS1
        w1Y = self.theta * self.rhoS2

        wX_p = w1X**2 / self.k1 + (w1X * w1Y) / self.k2
        wY_p = w1Y**2 / self.k2 + (w1X * w1Y) / self.k1

        wXX_pp = -(w1X**2) / self.k1
        wYY_pp = -(w1Y**2) / self.k2
        wXY_pp = -(w1X * w1Y) / self.k1 - (w1X * w1Y) / self.k2

        c1 = (
            -(w1X**2 / self.k1) * _func_K(self.k1 * T)
            - (w1Y**2 / self.k2) * _func_K(self.k2 * T)
            + (w1X**2 / self.k1 + 2 * (w1X * w1Y) / (self.k2 - self.k1))
            * _func_J(self.k1 * T)
            + (w1Y**2 / self.k2 - 2 * (w1X * w1Y) / (self.k2 - self.k1))
            * _func_J(self.k2 * T)
        )
        c2 = (
            wX_p * _func_J(self.k1 * T)
            + wY_p * _func_J(self.k2 * T)
            + wXX_pp * _func_J(2 * self.k1 * T)
            + wYY_pp * _func_J(2 * self.k2 * T)
            + wXY_pp * _func_J((self.k1 + self.k2) * T)
        )

        return self.alpha**2 * self.w**2 * self.xi0_0**2 * T**2 * (0.5 * c1 + c2)

    def _func_D(self, T):
        out = (1 - self.theta) * self.rhoS1 * _func_I(self.k1 * T)
        out += self.theta * self.rhoS2 * _func_I(self.k2 * T)
        return out

    def _func_F(self, T):
        out = (1 - self.theta) * self.rhoS1 * (1 - _func_I(self.k1 * T)) / (self.k1 * T)
        out += self.theta * self.rhoS2 * (1 - _func_I(self.k2 * T)) / (self.k2 * T)
        return out

    def _func_G(self, T):
        term1 = (
            (1 - self.theta) ** 2
            * (self.rhoS1 / self.k1) ** 2
            * (
                1
                + 0.5 * np.exp(-self.k1 * T) * (np.exp(-self.k1 * T) - 3 - self.k1 * T)
            )
        )
        term2 = (
            self.theta
            * (1 - self.theta)
            * self.rhoS1
            * self.rhoS2
            / (self.k1 * self.k2)
            * (
                -1
                + np.exp(-(self.k1 + self.k2) * T)
                + (
                    (2 * self.k2 - self.k1) * (1 - np.exp(-self.k1 * T))
                    - (2 * self.k1 - self.k2) * (1 - np.exp(-self.k2 * T))
                )
                / (self.k2 - self.k1)
            )
        )
        term3 = (
            self.theta**2
            * (self.rhoS2 / self.k2) ** 2
            * (
                1
                + 0.5 * np.exp(-self.k2 * T) * (np.exp(-self.k2 * T) - 3 - self.k2 * T)
            )
        )
        return self.alpha**2 * self.w**2 * self.xi0_0 * (term1 + term2 + term3)

    def _beta_1_bg_flat(self, T):
        return 0.5 * self.alpha * self.w * self._func_D(T)

    def _beta_2_bg_flat(self, T):
        return 0.25 * (
            self._func_G(T) / (T * self.xi0_0**0.5)
            - 0.5
            * self.alpha**2
            * self.w**2
            * self._func_D(T)
            * T
            * self.xi0_0**0.5
            * self._func_F(T)
        )

    def _atm_skew_1_bg_flat(self, T):
        return 0.5 * self.alpha * self.w * self._func_F(T)

    def _atm_skew_2_bg_flat(self, T) -> float | np.ndarray:
        _sig_vs = self.xi0_0**0.5
        num = (
            4 * self._c_mu_bg_flat(T) * T * _sig_vs**2
            - 3 * self._c_x_xi_bg_flat(T) ** 2
        )
        den = 8 * T**3 * _sig_vs**5
        return num / den

    def _ssr_0_bg_flat(self, T):
        num = (1 - self.theta) * self.rhoS1 * _func_I(
            self.k1 * T
        ) + self.theta * self.rhoS2 * _func_I(self.k2 * T)
        denom = (1 - self.theta) * self.rhoS1 * (1 - _func_I(self.k1 * T)) / (
            self.k1 * T
        ) + self.theta * self.rhoS2 * (1 - _func_I(self.k2 * T)) / (self.k2 * T)
        return num / denom

    def _ssr_1_vs_bg_flat(self, T):
        return (
            -self._beta_1_bg_flat(T)
            * self._atm_skew_2_bg_flat(T)
            / self._atm_skew_1_bg_flat(T) ** 2
        )

    def _ssr_1_bg_flat(self, T):
        return self._ssr_1_vs_bg_flat(T) + 0.5 * (
            self._func_G(T)
            / (self.alpha * self.w * self._func_F(T) * T * self.xi0_0**0.5)
            - 0.5 * self.alpha * self.w * self._func_D(T) * T * self.xi0_0**0.5
        )


def get_params_two_factor_bergomi(id: int):
    """Get Two-Factor Bergomi parameters for a given id."""
    if id not in [1, 2]:
        raise ValueError("Invalid id. Please choose a valid id.")
    if id == 1:
        # Bergomi (2016) - Stochastic Volatility Modeling - Table 8.2
        params = {
            "w": 2 * 1.74,
            "k1": 5.25,
            "k2": 0.28,
            "theta": 0.245,
            "rho12": 0.0,
            "rhoS1": -0.759,
            "rhoS2": -0.487,
        }
        xi0 = lambda t: 0.2**2 * np.ones_like(t)
    if id == 2:
        # Bergomi (2016) - Stochastic Volatility Modeling - Figure 9.2
        params = {
            "w": 2 * 1.74,
            "k1": 5.25,
            "k2": 0.28,
            "theta": 0.245,
            "rho12": 0.0,
            "rhoS1": -0.35,
            "rhoS2": -0.83,
        }
        xi0 = lambda t: 0.2**2 * np.ones_like(t)
    return params, xi0


@vectorize([float64(float64)], nopython=True, cache=True)
def _func_I(x):
    if x != 0.0:
        return -np.expm1(-x) / x
    return 1.0


@vectorize([float64(float64)], nopython=True, cache=True)
def _func_J(x):
    return (x - 1 + np.exp(-x)) / x**2


@vectorize([float64(float64)], nopython=True, cache=True)
def _func_K(x):
    return (1 - np.exp(-x) - x * np.exp(-x)) / x**2


@vectorize([float64(float64)], nopython=True, cache=True)
def _func_H(x):
    func_j = (x - 1 + np.exp(-x)) / x**2
    func_k = (1 - np.exp(-x) - x * np.exp(-x)) / x**2
    return (func_j - func_k) / x


@vectorize([float64(float64)], nopython=True, cache=True)
def _func_H_new(x):
    return np.exp(-x) / x


def _ou_factor_paths(dx, decay, x0):
    return _ou_factor_paths_jit(dx, decay, x0)


@njit(cache=True)
def _ou_factor_paths_jit(dx, decay, x0):
    n_disc, n_mc = dx.shape
    x = np.empty((n_disc + 1, n_mc), dtype=dx.dtype)
    x[0, :] = x0
    for i in range(n_disc):
        x[i + 1, :] = decay * x[i, :] + dx[i, :]
    return x


def exp_decay_kernel_matrices(k1, k2, n_disc, dt, dtype=np.float64):
    """
    Build lower-triangular exponential convolution matrices for two decay rates.

    Each matrix C satisfies:
        C[i, j] = exp(-k * dt * (i - j))  for j <= i, else 0

    Used to discretize convolution with exponential kernels (e.g. OU/Bergomi factors).
    """
    i = np.arange(n_disc, dtype=dtype)[:, None]
    j = np.arange(n_disc, dtype=dtype)[None, :]

    diff = i - j
    mask = diff >= 0

    C_x1 = np.where(mask, np.exp(-k1 * dt * diff), 0.0)
    C_x2 = np.where(mask, np.exp(-k2 * dt * diff), 0.0)

    return C_x1, C_x2
