# CDC Paper II referee code

This repository contains the lightweight numerical code and small support data expected alongside the Physics of the Dark Universe Paper II draft. It is intentionally limited to referee-facing reproducibility materials.

## Included

- `cdc_upgrade/python/cdc_boltzmann_reference.py`: stable reference solver for the CDC background and quasi-static growth checks.
- `cdc_upgrade/mcmc/`: Cobaya/CAMB approximation wrapper and run helpers for background-consistent likelihood experiments.
- `cdc_upgrade/postprocess/`: chain-table, convergence, growth, and SVG post-processing helpers.
- `data/cdc_perturbation_data.json`: small perturbation data file used in the draft-level figures/tables.

## Excluded

The repository deliberately excludes manuscript source/PDF files, LaTeX build products, large MCMC chain outputs, installed likelihood packages, caches, and temporary render images.

## Install

```bash
python -m pip install -r requirements.txt
```

Cobaya likelihood data and CAMB/Cobaya runtime setup are external dependencies and should be installed following Cobaya documentation.

## Typical smoke checks

```bash
python cdc_upgrade/python/cdc_boltzmann_reference.py --help
python cdc_upgrade/postprocess/plot_cdc_results.py --help
python cdc_upgrade/postprocess/cdc_growth_from_chain.py --help
```

## Manuscript note

For the journal manuscript, add a short Data and Code Availability statement once the public repository URL is final. Do not cite a placeholder URL.
