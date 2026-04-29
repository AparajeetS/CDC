#!/usr/bin/env python3
"""Run a full-likelihood Cobaya MCMC for the CDC background approximation."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def build_info(
    output_root: str,
    max_samples: int | None = None,
    covmat: str | None = None,
    seed: int | None = None,
    rminus1_stop: float = 0.02,
    rminus1_cl_stop: float = 0.2,
):
    try:
        from cdc_upgrade.mcmc.baseline_common import (
            base_cosmology_params,
            build_info as build_baseline_info,
            default_camb_args,
        )
        from cdc_upgrade.mcmc.cdc_camb_approx import CDCCAMBApprox
    except ModuleNotFoundError as exc:
        if exc.name != "cdc_upgrade":
            raise
        from baseline_common import (  # type: ignore[no-redef]
            base_cosmology_params,
            build_info as build_baseline_info,
            default_camb_args,
        )
        from cdc_camb_approx import CDCCAMBApprox  # type: ignore[no-redef]

    params_block = base_cosmology_params(theory_provides_omega_m=True)
    params_block.update(
        {
            "cdc_v": {
                "prior": {"min": 0.02, "max": 1.0},
                "ref": {"dist": "norm", "loc": 0.15, "scale": 0.03},
                "proposal": 0.03,
                "latex": r"v/M_\mathrm{Pl}",
            },
            "cdc_f_vac": {
                "prior": {"min": 0.0, "max": 0.5},
                "ref": {"dist": "norm", "loc": 0.12, "scale": 0.04},
                "proposal": 0.03,
                "latex": r"f_\mathrm{vac}",
            },
            "cdc_Omega_star": {
                "prior": {"min": 0.02, "max": 2.5},
                "ref": {"dist": "norm", "loc": 0.6, "scale": 0.12},
                "proposal": 0.08,
                "latex": r"\Omega_\ast",
            },
            "cdc_chi_ratio": {
                "prior": {"min": 0.0, "max": 1.0},
                "ref": {"dist": "norm", "loc": 0.2, "scale": 0.08},
                "proposal": 0.05,
                "latex": r"\chi_\mathrm{init}/v",
            },
            "cdc_bind_amp_bg": {
                "prior": {"min": 0.0, "max": 6.0},
                "ref": {"dist": "norm", "loc": 1.5, "scale": 0.5},
                "proposal": 0.25,
                "latex": r"A_\mathrm{bg}",
            },
            "cdc_bind_amp_env": {
                "prior": {"min": 0.0, "max": 6.0},
                "ref": {"dist": "norm", "loc": 1.0, "scale": 0.35},
                "proposal": 0.2,
                "latex": r"A_\mathrm{env}",
            },
            "cdc_bind_zhalf": {
                "prior": {"min": 0.05, "max": 5.0},
                "ref": {"dist": "norm", "loc": 1.0, "scale": 0.35},
                "proposal": 0.18,
                "latex": r"z_{1/2}^\mathrm{bind}",
            },
            "cdc_chi_ini": {
                "value": "lambda cdc_chi_ratio, cdc_v: cdc_chi_ratio*cdc_v",
                "latex": r"\chi_\mathrm{init}",
            },
            "cdc_w0": {"latex": r"w_0^\mathrm{CDC}"},
            "cdc_wa": {"latex": r"w_a^\mathrm{CDC}"},
            "cdc_lambda_eff": {"latex": r"\Lambda_\mathrm{eff}^\mathrm{CDC}"},
            "cdc_chi0": {"latex": r"\chi_0^\mathrm{CDC}"},
            "cdc_K0": {"latex": r"K_0^\mathrm{CDC}"},
        }
    )
    theory_block = {
        "cdc_approx": {
            "external": CDCCAMBApprox,
            "stop_at_error": False,
            "extra_args": {
                **default_camb_args(),
                "dark_energy_model": "DarkEnergyPPF",
            },
        }
    }
    return build_baseline_info(
        output_root=output_root,
        theory_block=theory_block,
        params_block=params_block,
        max_samples=max_samples,
        covmat=covmat,
        seed=seed,
        rminus1_stop=rminus1_stop,
        rminus1_cl_stop=rminus1_cl_stop,
    )


def main():
    parser = argparse.ArgumentParser(description="Run CDC CAMB approximate full-likelihood MCMC.")
    parser.add_argument(
        "--output-root",
        default=str(_ROOT / "cdc_upgrade" / "chains" / "cdc_camb_approx"),
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--covmat", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-changes", action="store_true")
    parser.add_argument("--rminus1-stop", type=float, default=0.02)
    parser.add_argument("--rminus1-cl-stop", type=float, default=0.2)
    args = parser.parse_args()

    try:
        from cobaya.run import run
    except ModuleNotFoundError as exc:
        if exc.name == "cobaya":
            raise SystemExit(
                "Cobaya is required to run this script. Install Cobaya and its "
                "likelihood data, then rerun the command."
            ) from exc
        raise

    gcc_bin = Path("C:/Users/apara/gcc/bin")
    if gcc_bin.exists():
        os.environ["PATH"] = str(gcc_bin) + os.pathsep + os.environ.get("PATH", "")

    info = build_info(
        args.output_root,
        max_samples=args.max_samples,
        covmat=args.covmat,
        seed=args.seed,
        rminus1_stop=args.rminus1_stop,
        rminus1_cl_stop=args.rminus1_cl_stop,
    )
    run(
        info,
        resume=args.resume,
        force=args.force,
        allow_changes=args.allow_changes,
    )


if __name__ == "__main__":
    main()
