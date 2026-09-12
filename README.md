# Skew-Stickiness Ratio (SSR)

[![CI](https://github.com/fbourgey/ssr/actions/workflows/ci.yml/badge.svg)](https://github.com/fbourgey/ssr/actions/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/package%20manager-uv-6340ac.svg)](https://docs.astral.sh/uv/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Python research code for reproducing skew-stickiness ratio (SSR) results in stochastic volatility models. Covers Heston, rough Heston, rough Bergomi, and two-factor Bergomi models.

## References

- Bourgey, F., Delemotte, J., & De Marco, S. (2026). *Refined expansions of the skew-stickiness ratio in stochastic volatility models*. [**Quantitative Finance**](https://doi.org/10.1080/14697688.2026.2714860), 1–18. Taylor & Francis. [SSRN 5387754](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5387754).
- Bourgey, F., De Marco, S., & Delemotte, J. (2024). *Smile Dynamics and Rough Volatility*. [SSRN 4911186](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4911186).

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

Model code and calibrated inputs are in [`src/ssr/`](src/ssr/). Generated plots are stored under [`figures/`](figures/).

## License

[MIT](LICENSE).
