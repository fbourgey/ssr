# Skew-Stickiness Ratio (SSR)

[![CI](https://github.com/fbourgey/ssr/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/ssr/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-6340ac.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python research code for reproducing skew-stickiness ratio (SSR) results in stochastic volatility models. The implementation covers Heston, rough Heston, rough Bergomi, and two-factor Bergomi models, with notebooks, tests, calibrated inputs, and generated figures.

## References

This repository reproduces results from:

- Bourgey, F., Delemotte, J., & De Marco, S. (2025). _Refined Expansions of the Skew-Stickiness Ratio in Stochastic Volatility Models_. Available at [SSRN 5387754](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5387754).

- Bourgey, F., De Marco, S., & Delemotte, J. (2024). _Smile Dynamics and Rough Volatility_. Available at [SSRN 4911186](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4911186).

## Setup

The project uses [uv](https://docs.astral.sh/uv/) and requires Python 3.12 or newer. To match CI exactly, install from the lockfile:

```bash
uv sync --locked
```

Useful commands:

```bash
uv run pytest
uv run pytest tests/test_rough_heston.py
uvx ruff@0.15.9 format --check .
uvx ruff@0.15.9 format .
uv run jupyter lab
```

## Reproducing Results

Open the notebooks from the repository root with Jupyter. The `ssr_expansions_*` notebooks reproduce results from _Refined Expansions of the Skew-Stickiness Ratio in Stochastic Volatility Models_; `smile_dynamics_and_rough_vol.ipynb` reproduces results from _Smile Dynamics and Rough Volatility_.

| Notebook | Purpose |
| --- | --- |
| `ssr_expansions_heston.ipynb` | Heston SSR expansion experiments and figures |
| `ssr_expansions_rheston.ipynb` | Rough Heston SSR expansion experiments and figures |
| `ssr_expansions_rbergomi.ipynb` | Rough Bergomi SSR expansion experiments and figures |
| `ssr_expansions_two_bergomi.ipynb` | Two-factor Bergomi SSR expansion experiments and figures |
| `smile_dynamics_and_rough_vol.ipynb` | Smile Dynamics and Rough Volatility paper |
| `xi0.ipynb` | Forward variance curve plots |
| `checks.ipynb` | Numerical consistency checks |

Generated plots are stored under `figures/`. Calibrated model inputs and forward variance curve helpers are defined in `data.py`.

## Repository Map

| File | Contents |
| --- | --- |
| `heston.py` | Heston model formulas |
| `rough_heston.py` | Rough Heston model formulas |
| `rough_bergomi.py` | Rough Bergomi model formulas |
| `two_bergomi.py` | Two-factor Bergomi model formulas |
| `model.py` | Shared `ForwardVarianceModel` base, SSR, implied-volatility, Monte Carlo, and Bergomi-Guyon helpers |
| `pade.py` | Padé approximation utilities |
| `utils.py` | Black pricing, implied volatility, quadrature, plotting, and simulation utilities |
| `data.py` | Calibrated parameters and forward variance curve helpers |
| `tests/` | Pytest coverage for formulas, configs, data, and utilities |
| `figures/` | Generated research figures |

## License

This project is distributed under the [MIT License](LICENSE).
