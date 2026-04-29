# CDC Paper II Referee Code

This repository contains the lightweight numerical code and support data for
the Physics of the Dark Universe Paper II draft. It is intentionally limited
to referee-facing reproducibility materials.

## Included

- `cdc_upgrade/paper2_check_claims.py`: checks the paper-facing numerical
  claims against the committed data and, optionally, recomputes the DESI DR2 +
  compressed Planck geometric table.
- `cdc_upgrade/paper2_geometric_reproduce.py`: standalone reproduction of the
  background-level DESI DR2 BAO + compressed Planck fit used for Table 3 in the
  draft.
- `cdc_upgrade/python/cdc_boltzmann_reference.py`: stable reference solver for
  the CDC background and quasi-static growth checks.
- `cdc_upgrade/mcmc/`: Cobaya/CAMB approximation wrapper and run helpers for
  background-consistent likelihood experiments.
- `cdc_upgrade/postprocess/`: chain-table, convergence, growth, and SVG
  post-processing helpers.
- `data/cdc_geometric_results.json`: geometric BAO+CMB table quoted in the
  paper.
- `data/cdc_perturbation_data.json`: perturbation data used in the draft-level
  figures and tables.

## Excluded

The repository deliberately excludes manuscript source/PDF files, LaTeX build
products, large MCMC chain outputs, installed likelihood packages, caches, and
temporary render images. The Paper III full CLASS/CAMB perturbation backend is
not part of this Paper II repository.

## Install

```bash
python -m pip install -r requirements.txt
```

Cobaya likelihood data and CAMB/Cobaya runtime setup are external dependencies
and should be installed following Cobaya documentation.

## Typical Smoke Checks

```bash
python cdc_upgrade/python/cdc_boltzmann_reference.py --help
python cdc_upgrade/postprocess/plot_cdc_results.py --help
python cdc_upgrade/postprocess/cdc_growth_from_chain.py --help
python cdc_upgrade/paper2_check_claims.py
```

## Full Paper II Number Check

The full geometric minimization takes longer than the smoke checks:

```bash
python cdc_upgrade/paper2_check_claims.py --recompute-geometric
```

This recomputes the DESI DR2 Table-IV BAO + compressed Planck 2018
distance-prior table and checks it against `data/cdc_geometric_results.json`.
