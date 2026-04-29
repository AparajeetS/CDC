#!/usr/bin/env python3
"""Shared Cobaya configuration helpers for matched CDC baseline comparisons."""

from __future__ import annotations

from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]


def packages_path() -> str:
    return str(_ROOT / "packages_test")


def default_camb_args() -> dict:
    return {
        "halofit_version": "mead",
        "lmax": 2600,
        "lens_potential_accuracy": 1,
        "num_massive_neutrinos": 1,
        "mnu": 0.06,
        "nnu": 3.044,
    }


def common_likelihood_block() -> dict:
    return {
        "planck_2018_lowl.TT": None,
        "planck_2018_lowl.EE": None,
        "planck_2018_highl_CamSpec.TTTEEE": None,
        "planck_2018_lensing.native": None,
        "bao.desi_dr2": None,
        "sn.pantheonplus": None,
    }


def common_sampler_block(
    max_samples: int | None = None,
    covmat: str | None = None,
    seed: int | None = None,
    rminus1_stop: float = 0.02,
    rminus1_cl_stop: float = 0.2,
) -> dict:
    block = {
        "mcmc": {
            "drag": True,
            "oversample_power": 0.4,
            "proposal_scale": 1.9,
            "covmat": covmat or "auto",
            "learn_proposal": True,
            "Rminus1_stop": float(rminus1_stop),
            "Rminus1_cl_stop": float(rminus1_cl_stop),
        }
    }
    if max_samples is not None:
        block["mcmc"]["max_samples"] = int(max_samples)
    if seed is not None:
        block["mcmc"]["seed"] = int(seed)
    return block


def base_cosmology_params(theory_provides_omega_m: bool = False) -> dict:
    omega_m_param = (
        {"latex": r"\Omega_\mathrm{m}"}
        if theory_provides_omega_m
        else {
            "derived": "lambda H0, ombh2, omch2, mnu: (ombh2 + omch2 + mnu/93.14)/((H0/100.)**2)",
            "latex": r"\Omega_\mathrm{m}",
        }
    )
    return {
        "logA": {
            "prior": {"min": 1.61, "max": 3.91},
            "ref": {"dist": "norm", "loc": 3.045, "scale": 0.01},
            "proposal": 0.005,
            "drop": True,
            "latex": r"\log(10^{10} A_\mathrm{s})",
        },
        "As": {
            "value": "lambda logA: 1e-10*np.exp(logA)",
            "latex": r"A_\mathrm{s}",
        },
        "ns": {
            "prior": {"min": 0.90, "max": 1.05},
            "ref": {"dist": "norm", "loc": 0.965, "scale": 0.004},
            "proposal": 0.002,
            "latex": r"n_\mathrm{s}",
        },
        "H0": {
            "prior": {"min": 55.0, "max": 85.0},
            "ref": {"dist": "norm", "loc": 68.0, "scale": 2.0},
            "proposal": 0.8,
            "latex": r"H_0",
        },
        "ombh2": {
            "prior": {"min": 0.020, "max": 0.0245},
            "ref": {"dist": "norm", "loc": 0.0224, "scale": 0.00015},
            "proposal": 0.00012,
            "latex": r"\omega_\mathrm{b}",
        },
        "omch2": {
            "prior": {"min": 0.08, "max": 0.16},
            "ref": {"dist": "norm", "loc": 0.120, "scale": 0.003},
            "proposal": 0.0015,
            "latex": r"\omega_\mathrm{cdm}",
        },
        "tau": {
            "prior": {"min": 0.01, "max": 0.12},
            "ref": {"dist": "norm", "loc": 0.055, "scale": 0.006},
            "proposal": 0.003,
            "latex": r"\tau_\mathrm{reio}",
        },
        "mnu": {"value": 0.06, "drop": True},
        "Omega_m": omega_m_param,
        "sigma8": {"latex": r"\sigma_8"},
        "S8": {
            "derived": "lambda sigma8, Omega_m: sigma8*np.sqrt(Omega_m/0.3)",
            "latex": r"S_8",
        },
    }


def build_info(
    output_root: str,
    theory_block: dict,
    params_block: dict,
    max_samples: int | None = None,
    covmat: str | None = None,
    seed: int | None = None,
    rminus1_stop: float = 0.02,
    rminus1_cl_stop: float = 0.2,
) -> dict:
    return {
        "packages_path": packages_path(),
        "output": output_root,
        "theory": theory_block,
        "likelihood": common_likelihood_block(),
        "params": params_block,
        "sampler": common_sampler_block(
            max_samples=max_samples,
            covmat=covmat,
            seed=seed,
            rminus1_stop=rminus1_stop,
            rminus1_cl_stop=rminus1_cl_stop,
        ),
    }
