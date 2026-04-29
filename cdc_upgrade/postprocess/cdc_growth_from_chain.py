#!/usr/bin/env python3
"""Post-process Cobaya CDC chains into perturbation-level growth observables.

This module does not modify CAMB. It reads existing Cobaya chain files,
reconstructs the CDC background from the sampled CDC parameters when available,
and solves an approximate linear growth equation with a conservative CDC
correction

    mu(a) = 1 + epsilon(a),

where epsilon(a) is sourced by the time variation of the CDC kinetic factor
through kappa(a) together with the rolling field amplitude chi_N(a). In the
CDC branch, the kinetic factor is driven by a structure-aware binding proxy
B(a) rather than plain rho_m(a).

If the chain does not contain the CDC microscopic parameters, the module falls
back to a CPL-like w(a)=w0+wa(1-a) background and uses a more conservative
late-time epsilon(a) based on the dark-energy rolling fraction.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cdc_upgrade.postprocess.simple_svg import write_three_panel_svg
from cdc_upgrade.python.cdc_boltzmann_reference import CDCParams, CDCReferenceModel


CHAIN_FILE_RE = re.compile(r"^(?P<stem>.+)\.(?P<index>\d+)\.txt$")
DEFAULT_OMEGA_R = 9.236e-5
DEFAULT_BARYON_FRACTION = 0.158


@dataclass
class ChainTable:
    names: list[str]
    values: np.ndarray

    @property
    def size(self) -> int:
        return int(self.values.shape[0])

    @property
    def weights(self) -> np.ndarray:
        return self.column("weight")

    @property
    def minuslogpost(self) -> np.ndarray:
        return self.column("minuslogpost")

    def column(self, name: str) -> np.ndarray:
        return self.values[:, self.names.index(name)]

    def has(self, name: str) -> bool:
        return name in self.names

    def row_dict(self, index: int) -> dict[str, float]:
        return {name: float(self.values[index, i]) for i, name in enumerate(self.names)}

    def bestfit_index(self) -> int:
        return int(np.argmin(self.minuslogpost))


def discover_chain_files(chain_root: Path) -> list[Path]:
    if chain_root.is_file():
        return [chain_root]

    parent = chain_root.parent
    stem = chain_root.name
    matches: list[tuple[int, Path]] = []
    for path in parent.glob(f"{stem}.*.txt"):
        match = CHAIN_FILE_RE.match(path.name)
        if match and match.group("stem") == stem:
            matches.append((int(match.group("index")), path))
    if not matches:
        raise FileNotFoundError(f"No Cobaya chain files found for root '{chain_root}'.")
    return [path for _, path in sorted(matches)]


def parse_chain_file(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"Chain file '{path}' is empty.")
    header = lines[0].lstrip("#").split()
    rows: list[list[float]] = []
    width = len(header)
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != width:
            continue
        try:
            rows.append([float(part) for part in parts])
        except ValueError:
            continue
    return header, rows


def load_chain_table(chain_root: Path) -> ChainTable:
    files = discover_chain_files(chain_root)
    names: list[str] | None = None
    rows: list[list[float]] = []
    for path in files:
        file_names, file_rows = parse_chain_file(path)
        if names is None:
            names = file_names
            rows.extend(file_rows)
            continue
        if file_names == names:
            rows.extend(file_rows)
            continue
        if set(file_names) != set(names):
            raise RuntimeError(f"Header mismatch in chain file '{path}'.")
        remap = [file_names.index(name) for name in names]
        rows.extend([[row[index] for index in remap] for row in file_rows])
    if names is None or not rows:
        raise RuntimeError(f"No valid samples were loaded from '{chain_root}'.")
    return ChainTable(names=names, values=np.asarray(rows, dtype=np.float64))


def weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    mean = float(np.average(values, weights=weights))
    var = float(np.average((values - mean) ** 2, weights=weights))
    return mean, math.sqrt(max(var, 0.0))


def get_value(row: dict[str, float], *names: str, default: float | None = None) -> float | None:
    for name in names:
        if name in row:
            return row[name]
    return default


def infer_omega_m(row: dict[str, float], omega_r: float = DEFAULT_OMEGA_R) -> float:
    value = get_value(row, "Omega_m")
    if value is not None:
        return float(value)
    H0 = float(get_value(row, "H0", default=67.5))
    h = H0 / 100.0
    ombh2 = float(get_value(row, "ombh2", default=0.0224))
    omch2 = float(get_value(row, "omch2", default=0.12))
    omnuh2 = float(get_value(row, "omnuh2", default=0.0))
    mnu = float(get_value(row, "mnu", default=0.06))
    if omnuh2 <= 0.0 and h > 0.0:
        omnuh2 = mnu / 93.14
    return (ombh2 + omch2 + omnuh2) / (h * h)


def infer_omega_b(row: dict[str, float], omega_m: float) -> float:
    H0 = float(get_value(row, "H0", default=67.5))
    h = H0 / 100.0
    ombh2 = get_value(row, "ombh2")
    if ombh2 is not None and h > 0.0:
        return float(ombh2) / (h * h)
    return min(DEFAULT_BARYON_FRACTION * omega_m, 0.9 * omega_m)


def build_cdc_background(
    row: dict[str, float],
    z_max: float,
    samples: int,
    omega_r: float,
) -> dict[str, np.ndarray | float]:
    H0 = float(get_value(row, "H0", default=67.5))
    h = H0 / 100.0
    omega_m = infer_omega_m(row, omega_r=omega_r)
    omega_b = infer_omega_b(row, omega_m)
    sigma8_0 = float(get_value(row, "sigma8", default=0.811))
    cdc_v = float(get_value(row, "cdc_v", default=0.15))
    cdc_f_vac = float(get_value(row, "cdc_f_vac", default=0.12))
    cdc_omega_star = float(get_value(row, "cdc_Omega_star", default=0.6))
    cdc_bind_amp_bg = float(get_value(row, "cdc_bind_amp_bg", "cdc_bind_amp", default=0.0))
    cdc_bind_amp_env = float(get_value(row, "cdc_bind_amp_env", default=1.0))
    cdc_bind_zhalf = float(get_value(row, "cdc_bind_zhalf", default=1.0))
    chi_init = get_value(row, "cdc_chi_ini")
    if chi_init is None:
        chi_ratio = float(get_value(row, "cdc_chi_ratio", default=0.2))
        chi_init = chi_ratio * cdc_v

    params = CDCParams(
        Omega_m=omega_m,
        Omega_b=omega_b,
        h=h,
        sigma8_0=sigma8_0,
        v=cdc_v,
        f_vac=cdc_f_vac,
        Omega_star=cdc_omega_star,
        chi_init=float(chi_init),
        bind_amp_bg=cdc_bind_amp_bg,
        bind_amp_env=cdc_bind_amp_env,
        bind_zhalf=cdc_bind_zhalf,
        Omega_r=omega_r,
        a_eval_min=min(1.0e-4, 1.0 / (1.0 + max(z_max, 5.0))),
        z_max_output=max(z_max, 5.0),
    )
    model = CDCReferenceModel(params)
    bg = model.solve_background(samples=samples)
    omega_m_a = params.Omega_m * bg["a"] ** -3 / bg["E2"]
    omega_r_a = params.Omega_r * bg["a"] ** -4 / bg["E2"]
    omega_de_a = 1.0 - omega_m_a - omega_r_a

    return {
        "mode": "cdc",
        "params": params,
        "a": bg["a"],
        "z": bg["z"],
        "E": bg["E"],
        "H": params.H0 * bg["E"],
        "w": bg["w"],
        "Omega_m_a": omega_m_a,
        "Omega_r_a": omega_r_a,
        "Omega_de_a": omega_de_a,
        "dlnH_dN": bg["dlnE_dN"],
        "K": bg["K"],
        "kappa": bg["kappa"],
        "chi_N": bg["chi_N"],
        "rho_m": bg["rho_m"],
        "rhoB": bg["rhoB"],
        "F_bind": bg["F_bind"],
        "structure_boost": bg["structure_boost"],
        "sigma8_0": params.sigma8_0,
        "env_bias": params.env_bias,
        "env_response": params.bind_amp_env * params.env_bias,
    }


def build_cpl_background(
    row: dict[str, float],
    z_max: float,
    samples: int,
    omega_r: float,
) -> dict[str, np.ndarray | float]:
    H0 = float(get_value(row, "H0", default=67.5))
    h = H0 / 100.0
    omega_m = infer_omega_m(row, omega_r=omega_r)
    sigma8_0 = float(get_value(row, "sigma8", default=0.811))
    w0 = float(get_value(row, "cdc_w0", "w0", default=-1.0))
    wa = float(get_value(row, "cdc_wa", "wa", default=0.0))

    a_min = min(1.0e-4, 1.0 / (1.0 + max(z_max, 5.0)))
    a = np.geomspace(a_min, 1.0, samples)
    z = 1.0 / a - 1.0
    omega_de0 = 1.0 - omega_m - omega_r
    w = w0 + wa * (1.0 - a)
    de = omega_de0 * a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))
    E2 = omega_m * a ** -3 + omega_r * a ** -4 + de
    E = np.sqrt(E2)
    dlnH_dN = np.gradient(np.log(E), np.log(a), edge_order=2)
    omega_m_a = omega_m * a ** -3 / E2
    omega_r_a = omega_r * a ** -4 / E2
    omega_de_a = 1.0 - omega_m_a - omega_r_a

    return {
        "mode": "cpl",
        "params": {
            "Omega_m": omega_m,
            "h": h,
            "w0": w0,
            "wa": wa,
            "sigma8_0": sigma8_0,
        },
        "a": a,
        "z": z,
        "E": E,
        "H": H0 * E,
        "w": w,
        "Omega_m_a": omega_m_a,
        "Omega_r_a": omega_r_a,
        "Omega_de_a": omega_de_a,
        "dlnH_dN": dlnH_dN,
        "sigma8_0": sigma8_0,
    }


def reconstruct_background(
    row: dict[str, float],
    z_max: float,
    samples: int,
    omega_r: float = DEFAULT_OMEGA_R,
) -> dict[str, np.ndarray | float]:
    if all(key in row for key in ("cdc_v", "cdc_f_vac", "cdc_Omega_star")):
        return build_cdc_background(row, z_max=z_max, samples=samples, omega_r=omega_r)
    return build_cpl_background(row, z_max=z_max, samples=samples, omega_r=omega_r)


def approximate_mu(
    background: dict[str, np.ndarray | float],
    mu_scale: float,
    mu_clip: float,
) -> tuple[np.ndarray, np.ndarray]:
    if background["mode"] == "cdc":
        kappa = np.asarray(background["kappa"], dtype=np.float64)
        chi_N = np.asarray(background["chi_N"], dtype=np.float64)
        epsilon = mu_scale * kappa * chi_N * chi_N
    else:
        omega_de = np.clip(np.asarray(background["Omega_de_a"], dtype=np.float64), 0.0, 1.0)
        w = np.asarray(background["w"], dtype=np.float64)
        epsilon = 0.5 * mu_scale * omega_de * np.clip(1.0 + w, 0.0, None)
    epsilon = np.clip(epsilon, -abs(mu_clip), abs(mu_clip))
    return 1.0 + epsilon, epsilon


def solve_growth(
    background: dict[str, np.ndarray | float],
    mu_scale: float,
    mu_clip: float,
) -> dict[str, np.ndarray]:
    a = np.asarray(background["a"], dtype=np.float64)
    if np.any(np.diff(a) <= 0.0):
        raise RuntimeError("Background scale factor grid is not strictly increasing.")

    N = np.log(a)
    omega_m_a = np.asarray(background["Omega_m_a"], dtype=np.float64)
    dlnH_dN = np.asarray(background["dlnH_dN"], dtype=np.float64)
    mu, epsilon = approximate_mu(background, mu_scale=mu_scale, mu_clip=mu_clip)

    omega_interp = interp1d(N, omega_m_a, kind="cubic", bounds_error=False, fill_value="extrapolate")
    dlnH_interp = interp1d(N, dlnH_dN, kind="cubic", bounds_error=False, fill_value="extrapolate")
    mu_interp = interp1d(N, mu, kind="cubic", bounds_error=False, fill_value="extrapolate")

    def rhs(Nloc: float, y: np.ndarray) -> np.ndarray:
        D, U = y
        dD = U
        dU = -(2.0 + float(dlnH_interp(Nloc))) * U + 1.5 * float(omega_interp(Nloc)) * float(mu_interp(Nloc)) * D
        return np.array([dD, dU], dtype=np.float64)

    D_ini = a[0]
    sol = solve_ivp(
        rhs,
        (float(N[0]), 0.0),
        y0=np.array([D_ini, D_ini], dtype=np.float64),
        method="DOP853",
        t_eval=N,
        rtol=1.0e-8,
        atol=1.0e-10,
        max_step=0.05,
    )
    if not sol.success:
        raise RuntimeError(f"Growth integration failed: {sol.message}")

    D_raw = sol.y[0]
    U = sol.y[1]
    D = D_raw / D_raw[-1]
    f = U / np.maximum(D_raw, 1.0e-30)
    fsigma8 = float(background["sigma8_0"]) * D * f

    return {
        "a": a,
        "z": 1.0 / a - 1.0,
        "D": D,
        "f": f,
        "fsigma8": fsigma8,
        "mu": mu,
        "epsilon": epsilon,
    }


def sample_curves_on_grid(
    background: dict[str, np.ndarray | float],
    growth: dict[str, np.ndarray],
    z_grid: np.ndarray,
) -> dict[str, np.ndarray]:
    a_grid = 1.0 / (1.0 + z_grid)
    out: dict[str, np.ndarray] = {"z": z_grid}
    for name, source in (
        ("H", np.asarray(background["H"], dtype=np.float64)),
        ("Omega_m_z", np.asarray(background["Omega_m_a"], dtype=np.float64)),
        ("w", np.asarray(background["w"], dtype=np.float64)),
        ("mu", np.asarray(growth["mu"], dtype=np.float64)),
        ("epsilon", np.asarray(growth["epsilon"], dtype=np.float64)),
        ("D", np.asarray(growth["D"], dtype=np.float64)),
        ("f", np.asarray(growth["f"], dtype=np.float64)),
        ("fsigma8", np.asarray(growth["fsigma8"], dtype=np.float64)),
    ):
        interp = interp1d(
            np.asarray(background["a"], dtype=np.float64),
            source,
            kind="cubic",
            bounds_error=False,
            fill_value="extrapolate",
        )
        out[name] = np.asarray(interp(a_grid), dtype=np.float64)
    return out


def summarize_parameter_stats(chain: ChainTable, names: list[str]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    weights = chain.weights
    for name in names:
        if not chain.has(name):
            continue
        mean, std = weighted_mean_std(chain.column(name), weights)
        stats[name] = {"mean": mean, "std": std}
    return stats


def posterior_draw_indices(chain: ChainTable, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probs = chain.weights / np.sum(chain.weights)
    return rng.choice(chain.size, size=n_draws, replace=True, p=probs)


def make_bands(curves: list[dict[str, np.ndarray]], fields: list[str]) -> dict[str, np.ndarray]:
    bands: dict[str, np.ndarray] = {}
    for field in fields:
        stack = np.stack([curve[field] for curve in curves], axis=0)
        bands[f"{field}_p16"] = np.percentile(stack, 16.0, axis=0)
        bands[f"{field}_p84"] = np.percentile(stack, 84.0, axis=0)
    return bands


def write_growth_outputs(
    output_dir: Path,
    best_curve: dict[str, np.ndarray],
    bands: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "z": best_curve["z"].tolist(),
        "H": best_curve["H"].tolist(),
        "Omega_m_z": best_curve["Omega_m_z"].tolist(),
        "w": best_curve["w"].tolist(),
        "mu": best_curve["mu"].tolist(),
        "epsilon": best_curve["epsilon"].tolist(),
        "D": best_curve["D"].tolist(),
        "f": best_curve["f"].tolist(),
        "fsigma8": best_curve["fsigma8"].tolist(),
    }
    for key, value in bands.items():
        payload[key] = value.tolist()
    (output_dir / "cdc_growth_predictions.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    panels = [
        {
            "title": "Expansion History",
            "xlabel": "z",
            "ylabel": "H(z) [km s^-1 Mpc^-1]",
            "lines": [
                {"x": best_curve["z"], "y": best_curve["H"], "label": "Best fit", "color": "#d97706"},
                {"x": best_curve["z"], "y": bands["H_p16"], "label": "16th pct", "color": "#fbbf24", "dash": "6,4"},
                {"x": best_curve["z"], "y": bands["H_p84"], "label": "84th pct", "color": "#b45309", "dash": "6,4"},
            ],
        },
        {
            "title": "Effective Growth Source",
            "xlabel": "z",
            "ylabel": "mu(z)",
            "lines": [
                {"x": best_curve["z"], "y": best_curve["mu"], "label": "Best fit", "color": "#2563eb"},
                {"x": best_curve["z"], "y": bands["mu_p16"], "label": "16th pct", "color": "#60a5fa", "dash": "6,4"},
                {"x": best_curve["z"], "y": bands["mu_p84"], "label": "84th pct", "color": "#1d4ed8", "dash": "6,4"},
            ],
        },
        {
            "title": "Growth Observable",
            "xlabel": "z",
            "ylabel": "f sigma_8(z)",
            "lines": [
                {"x": best_curve["z"], "y": best_curve["fsigma8"], "label": "Best fit", "color": "#059669"},
                {"x": best_curve["z"], "y": bands["fsigma8_p16"], "label": "16th pct", "color": "#34d399", "dash": "6,4"},
                {"x": best_curve["z"], "y": bands["fsigma8_p84"], "label": "84th pct", "color": "#047857", "dash": "6,4"},
            ],
        },
    ]
    write_three_panel_svg(output_dir / "cdc_growth_summary.svg", panels)


def load_growth_data(path: Path) -> dict[str, np.ndarray]:
    data = np.genfromtxt(path, names=True, delimiter=None, encoding="utf-8")
    names = {name.lower(): name for name in data.dtype.names or ()}
    try:
        z = np.asarray(data[names["z"]], dtype=np.float64)
        fsigma8 = np.asarray(data[names["fsigma8"]], dtype=np.float64)
    except KeyError as exc:
        raise RuntimeError("Growth data file must contain 'z' and 'fsigma8' columns.") from exc
    if "sigma" in names:
        sigma = np.asarray(data[names["sigma"]], dtype=np.float64)
    elif "err" in names:
        sigma = np.asarray(data[names["err"]], dtype=np.float64)
    else:
        raise RuntimeError("Growth data file must contain a 'sigma' or 'err' column.")
    return {"z": z, "fsigma8": fsigma8, "sigma": sigma}


def growth_data_summary(curve: dict[str, np.ndarray], growth_data: dict[str, np.ndarray]) -> dict[str, float]:
    pred = np.interp(growth_data["z"], curve["z"], curve["fsigma8"])
    resid = pred - growth_data["fsigma8"]
    chi2 = float(np.sum((resid / growth_data["sigma"]) ** 2))
    return {"n_points": int(growth_data["z"].size), "chi2_bestfit": chi2}


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process CDC Cobaya chains into growth observables.")
    parser.add_argument("--chain-root", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("C:/Research/cdc_journal/cdc_upgrade/figures"),
    )
    parser.add_argument("--z-max", type=float, default=2.0)
    parser.add_argument("--nz", type=int, default=200)
    parser.add_argument("--background-samples", type=int, default=1200)
    parser.add_argument("--posterior-draws", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mu-scale", type=float, default=1.0)
    parser.add_argument("--mu-clip", type=float, default=0.25)
    parser.add_argument("--growth-data", type=Path, default=None)
    args = parser.parse_args()

    chain = load_chain_table(args.chain_root)
    z_grid = np.linspace(0.0, args.z_max, args.nz)

    best_row = chain.row_dict(chain.bestfit_index())
    best_background = reconstruct_background(
        best_row,
        z_max=args.z_max,
        samples=max(args.background_samples, args.nz + 50),
    )
    best_growth = solve_growth(best_background, mu_scale=args.mu_scale, mu_clip=args.mu_clip)
    best_curve = sample_curves_on_grid(best_background, best_growth, z_grid=z_grid)

    posterior_curves: list[dict[str, np.ndarray]] = [best_curve]
    draw_indices = posterior_draw_indices(chain, n_draws=max(args.posterior_draws, 1), seed=args.seed)
    for index in draw_indices:
        row = chain.row_dict(int(index))
        try:
            background = reconstruct_background(
                row,
                z_max=args.z_max,
                samples=max(args.background_samples, args.nz + 50),
            )
            growth = solve_growth(background, mu_scale=args.mu_scale, mu_clip=args.mu_clip)
        except Exception:
            continue
        posterior_curves.append(sample_curves_on_grid(background, growth, z_grid=z_grid))

    bands = make_bands(
        posterior_curves,
        fields=["H", "mu", "D", "f", "fsigma8", "Omega_m_z", "w", "epsilon"],
    )
    write_growth_outputs(args.output_dir, best_curve, bands)

    stats = summarize_parameter_stats(
        chain,
        names=[
            "H0",
            "Omega_m",
            "sigma8",
            "S8",
            "cdc_v",
            "cdc_f_vac",
            "cdc_Omega_star",
            "cdc_chi_ratio",
            "cdc_bind_amp_bg",
            "cdc_bind_amp_env",
            "cdc_bind_zhalf",
            "cdc_w0",
            "cdc_wa",
        ],
    )
    summary = {
        "chain_root": str(args.chain_root),
        "n_chain_rows": chain.size,
        "total_chain_weight": float(np.sum(chain.weights)),
        "n_successful_curves": len(posterior_curves),
        "bestfit_minuslogpost": float(best_row["minuslogpost"]),
        "bestfit_parameters": {
            key: float(best_row[key])
            for key in (
                "H0",
                "Omega_m",
                "sigma8",
                "S8",
                "cdc_v",
                "cdc_f_vac",
                "cdc_Omega_star",
                "cdc_chi_ratio",
                "cdc_bind_amp_bg",
                "cdc_bind_amp_env",
                "cdc_bind_zhalf",
                "cdc_w0",
                "cdc_wa",
            )
            if key in best_row
        },
        "weighted_parameter_stats": stats,
        "bestfit_observables": {
            "mu0": float(best_curve["mu"][0]),
            "fsigma8_0": float(best_curve["fsigma8"][0]),
            "f0": float(best_curve["f"][0]),
        },
    }
    if args.growth_data is not None:
        growth_data = load_growth_data(args.growth_data)
        summary["growth_data_comparison"] = growth_data_summary(best_curve, growth_data)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cdc_growth_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Loaded {chain.size} chain rows from {args.chain_root}")
    print(f"Best-fit sample mode: {best_background['mode']}")
    print(f"Successful posterior curves: {len(posterior_curves)}")
    print(f"f sigma_8(0) = {best_curve['fsigma8'][0]:.6f}")
    print(f"mu(0)        = {best_curve['mu'][0]:.6f}")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
