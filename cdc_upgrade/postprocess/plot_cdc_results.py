#!/usr/bin/env python3
"""Generate CDC publication figures from either reference solutions or Cobaya chains."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cdc_upgrade.postprocess.simple_svg import write_three_panel_svg
from cdc_upgrade.python.cdc_boltzmann_reference import CDCParams, CDCReferenceModel, lcdm_E


def plot_reference(output_dir: Path, k_hmpc: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = CDCReferenceModel(CDCParams())
    bg = model.solve_background()
    gr = model.solve_growth_qs(k_hmpc=k_hmpc)

    z = bg["z"]
    mask = z <= model.p.z_max_output
    z_plot = z[mask]

    H_cdc = model.p.H0 * bg["E"][mask]
    H_lcdm = model.p.H0 * lcdm_E(z_plot, model.p.Omega_m, model.p.Omega_r)

    z_gr = gr["z"]
    mask_gr = z_gr <= model.p.z_max_output
    write_three_panel_svg(
        output_dir / "cdc_reference_panels.svg",
        [
            {
                "title": "H(z) comparison",
                "xlabel": "z",
                "ylabel": "H(z) [km s^-1 Mpc^-1]",
                "lines": [
                    {"x": z_plot, "y": H_cdc, "label": "CDC", "color": "#d97706"},
                    {"x": z_plot, "y": H_lcdm, "label": "LambdaCDM", "color": "#2563eb", "dash": "8,5"},
                ],
            },
            {
                "title": "Equation of state",
                "xlabel": "z",
                "ylabel": "w(z)",
                "lines": [
                    {"x": z_plot, "y": bg["w"][mask], "label": "CDC", "color": "#059669"},
                    {"x": z_plot, "y": np.full_like(z_plot, -1.0), "label": "w=-1", "color": "#444444", "dash": "6,4"},
                ],
            },
            {
                "title": "Growth rate",
                "xlabel": "z",
                "ylabel": "f sigma_8(z)",
                "lines": [
                    {"x": z_gr[mask_gr], "y": gr["fsigma8"][mask_gr], "label": "CDC", "color": "#7c3aed"},
                ],
            },
        ],
    )


def plot_posteriors(chain_root: Path, output_dir: Path) -> None:
    try:
        from getdist import loadMCSamples, plots
    except ImportError as exc:
        raise RuntimeError("getdist is required for posterior contours.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    samples = loadMCSamples(str(chain_root))
    plotter = plots.get_subplot_plotter()
    plotter.settings.alpha_filled_add = 0.5
    plotter.triangle_plot(
        [samples],
        [
            "Omega_m",
            "H0",
            "cdc_v",
            "cdc_f_vac",
            "cdc_Omega_star",
            "cdc_bind_amp_bg",
            "cdc_bind_amp_env",
            "cdc_bind_zhalf",
        ],
        filled=True,
    )
    plotter.export(str(output_dir / "cdc_posterior_triangle.pdf"))


def main():
    parser = argparse.ArgumentParser(description="Generate CDC publication figures.")
    parser.add_argument("--output-dir", type=Path, default=Path("C:/Research/cdc_journal/cdc_upgrade/figures"))
    parser.add_argument("--chain-root", type=Path, default=None)
    parser.add_argument("--k-hmpc", type=float, default=0.1)
    args = parser.parse_args()

    plot_reference(args.output_dir, args.k_hmpc)
    if args.chain_root is not None:
        plot_posteriors(args.chain_root, args.output_dir)


if __name__ == "__main__":
    main()
