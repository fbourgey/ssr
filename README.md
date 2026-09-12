# Skew-Stickiness Ratio (SSR)

[![CI](https://github.com/fbourgey/ssr/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/ssr/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-6340ac.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python research code for reproducing skew-stickiness ratio (SSR) results in stochastic volatility models. Covers Heston, rough Heston, rough Bergomi, and two-factor Bergomi models.

## References

- Bourgey, F., Delemotte, J., & De Marco, S. (2026). *Refined expansions of the skew-stickiness ratio in stochastic volatility models*. [**Quantitative Finance**](https://doi.org/10.1080/14697688.2026.2714860), 1–18. Taylor & Francis. [SSRN 5387754](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5387754).
- Bourgey, F., De Marco, S., & Delemotte, J. (2024). *Smile Dynamics and Rough Volatility*. [SSRN 4911186](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4911186).
- Fukasawa, M. (2026). *On the Skew Stickiness Ratio*. [arXiv:2602.05241v2](https://arxiv.org/abs/2602.05241v2).

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). From the repository root:

```bash
uv sync --locked
uv run jupyter lab
```

To run the tests:

```bash
uv run pytest
```

## Reproducing Results

The `ssr_expansions_*` notebooks reproduce results from *Refined expansions of the skew-stickiness ratio in stochastic volatility models*; `smile_dynamics_and_rough_vol.ipynb` reproduces results from *Smile Dynamics and Rough Volatility*.

| Notebook | Description |
| --- | --- |
| [ssr_expansions_heston.ipynb](ssr_expansions_heston.ipynb) | Heston SSR expansions, calibrated examples, and approximation errors. |
| [ssr_expansions_rheston.ipynb](ssr_expansions_rheston.ipynb) | Rough Heston SSR expansions and numerical checks. |
| [ssr_expansions_rbergomi.ipynb](ssr_expansions_rbergomi.ipynb) | Rough Bergomi SSR expansions, approximation errors, and time-step sensitivity. |
| [ssr_expansions_two_bergomi.ipynb](ssr_expansions_two_bergomi.ipynb) | Two-factor Bergomi SSR expansions with benchmark and calibrated parameters. |
| [smile_dynamics_and_rough_vol.ipynb](smile_dynamics_and_rough_vol.ipynb) | Smile dynamics figures for Bergomi and rough Heston models, with market comparisons. |
| [xi0.ipynb](xi0.ipynb) | Calibrated initial forward variance curves for all models. |
| [checks.ipynb](checks.ipynb) | Numerical comparisons of SSR, ATM skew, and implied volatility methods. |
| [ssr_fukasawa.ipynb](ssr_fukasawa.ipynb) | Fukasawa's estimator versus finite differences for two-factor and rough Bergomi. |

Model code and calibrated inputs are in [`src/ssr/`](src/ssr/). Generated plots are stored under [`figures/`](figures/).

## License

[MIT](LICENSE).
