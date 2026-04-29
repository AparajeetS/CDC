#!/usr/bin/env python3
"""Create a LaTeX parameter table from Cobaya/GetDist chains."""

from __future__ import annotations

import argparse
from pathlib import Path


PARAMS = [
    ("Omega_m", r"\Omega_m"),
    ("H0", r"H_0"),
    ("cdc_v", r"v/M_{\rm Pl}"),
    ("cdc_f_vac", r"f_{\rm vac}"),
    ("cdc_Omega_star", r"\Omega_\ast"),
    ("cdc_chi_ini", r"\chi_{\rm init}"),
    ("cdc_bind_amp_bg", r"A_{\rm bg}"),
    ("cdc_bind_amp_env", r"A_{\rm env}"),
    ("cdc_bind_zhalf", r"z_{1/2}^{\rm bind}"),
    ("sigma8", r"\sigma_8"),
    ("S8", r"S_8"),
    ("cdc_w0", r"w_0^{\rm CDC}"),
    ("cdc_wa", r"w_a^{\rm CDC}"),
]


def main():
    parser = argparse.ArgumentParser(description="Build LaTeX parameter table from chains.")
    parser.add_argument("--chain-root", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("C:/Research/cdc_journal/cdc_upgrade/figures/cdc_parameter_table.tex"),
    )
    args = parser.parse_args()

    try:
        from getdist import loadMCSamples
    except ImportError as exc:
        raise RuntimeError("getdist is required to build the parameter table.") from exc

    samples = loadMCSamples(str(args.chain_root))
    args.output.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        r"\begin{tabular}{lcc}",
        r"\toprule",
        r"Parameter & Mean $\pm$ 68\% C.L. & Best-fit \\",
        r"\midrule",
    ]

    best = samples.getLikeStats().bestfit_sample
    par_names = {name.name: name.label for name in samples.paramNames.names}

    for name, latex in PARAMS:
        marg = samples.getMargeStats().parWithName(name)
        if marg is None:
            continue
        mean = marg.mean
        lower = mean - marg.limits[0].lower
        upper = marg.limits[0].upper - mean
        best_fit = getattr(best, name, float("nan"))
        lines.append(
            rf"${latex}$ & ${mean:.4g}_{{-{lower:.2g}}}^{{+{upper:.2g}}}$ & ${best_fit:.4g}$ \\"
        )

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    args.output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
