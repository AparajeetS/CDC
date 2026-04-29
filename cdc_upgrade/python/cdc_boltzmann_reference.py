#!/usr/bin/env python3
"""Reference CDC solver for background evolution and quasi-static growth.

This module is a numerically stable reference implementation for model
development, figure generation, and smoke tests. It is not a replacement
for a full Einstein-Boltzmann code.

The key numerical choice is the canonical momentum variable
    pi = a^3 K E dchi/dN
which avoids directly integrating a stiff dlnK/dN friction term.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar


_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


_C_KMS = 299792.458


@dataclass
class CDCParams:
    Omega_m: float = 0.31
    Omega_b: float = 0.049
    h: float = 0.68
    sigma8_0: float = 0.811
    v: float = 0.15
    f_vac: float = 0.20
    Omega_star: float = 0.60
    chi_init: float = 0.03
    n: float = 2.0
    beta_B: float = 1.0
    bind_amp_bg: float = 0.0
    bind_amp_env: float = 1.0
    bind_zhalf: float = 1.0
    bind_slope: float = 4.0
    env_bias: float = 1.0
    Omega_r: float = 9.236e-5
    a_ini: float = 1.0e-4
    a_eval_min: float = 1.0e-4
    z_max_output: float = 5.0

    @property
    def H0(self) -> float:
        return 100.0 * self.h

    @property
    def H0_over_c_Mpc(self) -> float:
        return self.H0 / _C_KMS

    @property
    def Omega_de_ref(self) -> float:
        return 1.0 - self.Omega_m - self.Omega_r


class CDCReferenceModel:
    def __init__(self, params: CDCParams):
        self.p = params
        self._background = None

    @staticmethod
    def _unwrap(value: np.ndarray | float) -> np.ndarray | float:
        array = np.asarray(value)
        if array.ndim == 0:
            return float(array)
        return array

    def delta_vac(self) -> float:
        return self.p.f_vac * self.p.Omega_de_ref

    def A(self) -> float:
        return self.delta_vac() / self.p.v**4

    def potential(self, chi: np.ndarray | float, lambda_eff: float) -> np.ndarray | float:
        return lambda_eff + self.A() * (chi * chi - self.p.v * self.p.v) ** 2

    def dV(self, chi: np.ndarray | float) -> np.ndarray | float:
        return 4.0 * self.A() * chi * (chi * chi - self.p.v * self.p.v)

    def ddV(self, chi: np.ndarray | float) -> np.ndarray | float:
        return 4.0 * self.A() * (3.0 * chi * chi - self.p.v * self.p.v)

    def matter_density(self, a: np.ndarray | float) -> np.ndarray | float:
        return self._unwrap(self.p.Omega_m * np.power(a, -3.0))

    def collapsed_fraction(self, a: np.ndarray | float) -> np.ndarray | float:
        a = np.asarray(a, dtype=np.float64)
        a_half = 1.0 / (1.0 + max(self.p.bind_zhalf, 1.0e-6))
        slope = max(self.p.bind_slope, 1.0e-3)
        ratio = np.power(np.maximum(a_half / np.maximum(a, 1.0e-12), 1.0e-12), slope)
        return self._unwrap(1.0 / (1.0 + ratio))

    def dFbind_dN(self, a: np.ndarray | float) -> np.ndarray | float:
        F = np.asarray(self.collapsed_fraction(a), dtype=np.float64)
        slope = max(self.p.bind_slope, 1.0e-3)
        return self._unwrap(slope * F * (1.0 - F))

    def env_response(self) -> float:
        return self.p.bind_amp_env * self.p.env_bias

    def structure_boost(self, a: np.ndarray | float) -> np.ndarray | float:
        return self._unwrap(1.0 + self.p.bind_amp_bg * self.collapsed_fraction(a))

    def dln_structure_boost_dN(self, a: np.ndarray | float) -> np.ndarray | float:
        F = np.asarray(self.collapsed_fraction(a), dtype=np.float64)
        dF = np.asarray(self.dFbind_dN(a), dtype=np.float64)
        boost = 1.0 + self.p.bind_amp_bg * F
        return self._unwrap(self.p.bind_amp_bg * dF / np.maximum(boost, 1.0e-30))

    def dlnrhoB_dN(self, a: np.ndarray | float) -> np.ndarray | float:
        return self._unwrap(-3.0 + np.asarray(self.dln_structure_boost_dN(a), dtype=np.float64))

    def environment_factor(self, a: np.ndarray | float, delta_env: np.ndarray | float = 0.0) -> np.ndarray | float:
        a = np.asarray(a, dtype=np.float64)
        delta_env = np.asarray(delta_env, dtype=np.float64)
        factor = 1.0 + self.env_response() * self.collapsed_fraction(a) * delta_env
        return self._unwrap(np.maximum(factor, 1.0e-3))

    def rhoB(self, a: np.ndarray | float, delta_env: np.ndarray | float = 0.0) -> np.ndarray | float:
        base = self.matter_density(a) * self.structure_boost(a)
        return self._unwrap(base * self.environment_factor(a, delta_env=delta_env))

    def dlnK_dlnB(self, a: np.ndarray | float) -> np.ndarray | float:
        ratio = np.maximum(self.rhoB(a) / self.p.Omega_star, 0.0)
        logx = self.p.n * np.log(np.maximum(ratio, 1.0e-300))
        x = np.exp(np.clip(logx, -700.0, 200.0))
        return self._unwrap(self.p.n * x / (1.0 + x))

    def kinetic_factor(self, a: np.ndarray | float) -> np.ndarray | float:
        ratio = np.maximum(self.rhoB(a) / self.p.Omega_star, 0.0)
        logx = self.p.n * np.log(np.maximum(ratio, 1.0e-300))
        x = np.exp(np.clip(logx, -700.0, 200.0))
        return self._unwrap(1.0 + x)

    def kappa(self, a: np.ndarray | float) -> np.ndarray | float:
        response = np.asarray(self.dlnK_dlnB(a), dtype=np.float64)
        F = np.asarray(self.collapsed_fraction(a), dtype=np.float64)
        return self._unwrap(self.p.beta_B * self.env_response() * F * response)

    def dlnK_dN(self, a: np.ndarray | float) -> np.ndarray | float:
        response = np.asarray(self.dlnK_dlnB(a), dtype=np.float64)
        dlnB = np.asarray(self.dlnrhoB_dN(a), dtype=np.float64)
        return self._unwrap(response * dlnB)

    def kinetic_density(self, a: np.ndarray, pi: np.ndarray, K: np.ndarray) -> np.ndarray:
        return 0.5 * pi * pi / (np.power(a, 6.0) * K)

    def _rhs_background(self, N: float, y: np.ndarray, lambda_eff: float) -> np.ndarray:
        a = np.exp(N)
        chi, pi = y
        K = self.kinetic_factor(a)
        E2 = (
            self.p.Omega_m * a ** -3
            + self.p.Omega_r * a ** -4
            + self.potential(chi, lambda_eff)
            + 0.5 * pi * pi / (a**6 * K)
        )
        if not np.isfinite(E2) or E2 <= 0.0:
            raise FloatingPointError("CDC background reached non-positive E^2.")
        E = np.sqrt(E2)
        return np.array(
            [
                pi / (a**3 * K * E),
                -(a**3) * self.dV(chi) / E,
            ]
        )

    def _integrate_background(self, lambda_eff: float):
        N_ini = np.log(self.p.a_ini)
        sol = solve_ivp(
            fun=lambda N, y: self._rhs_background(N, y, lambda_eff),
            t_span=(N_ini, 0.0),
            y0=np.array([self.p.chi_init, 0.0]),
            method="BDF",
            dense_output=True,
            rtol=1.0e-8,
            atol=1.0e-10,
            max_step=0.05,
        )
        if not sol.success:
            raise RuntimeError(f"Background integration failed: {sol.message}")
        return sol

    def _closure_residual(self, lambda_eff: float) -> float:
        sol = self._integrate_background(lambda_eff)
        chi_0 = sol.y[0, -1]
        pi_0 = sol.y[1, -1]
        K_0 = self.kinetic_factor(1.0)
        E2_0 = (
            self.p.Omega_m
            + self.p.Omega_r
            + self.potential(chi_0, lambda_eff)
            + 0.5 * pi_0 * pi_0 / K_0
        )
        return E2_0 - 1.0

    def solve_background(self, samples: int = 1600):
        if self._background is not None:
            return self._background

        bracket = None
        candidates = np.linspace(0.0, 1.5, 16)
        values = []
        for val in candidates:
            try:
                res = self._closure_residual(float(val))
                values.append((float(val), float(res)))
            except Exception:
                continue
        for (x0, f0), (x1, f1) in zip(values[:-1], values[1:]):
            if f0 == 0.0:
                bracket = (x0, x0 + 1.0e-6)
                break
            if f0 * f1 < 0.0:
                bracket = (x0, x1)
                break
        if bracket is None:
            raise RuntimeError(f"Could not bracket lambda_eff root from scan: {values}")

        root = root_scalar(self._closure_residual, bracket=bracket, method="brentq", xtol=1.0e-10)
        if not root.converged:
            raise RuntimeError("lambda_eff root solve failed.")

        lambda_eff = root.root
        sol = self._integrate_background(lambda_eff)
        N = np.linspace(np.log(self.p.a_eval_min), 0.0, samples)
        a = np.exp(N)
        chi = sol.sol(N)[0]
        pi = sol.sol(N)[1]
        K = self.kinetic_factor(a)
        V = self.potential(chi, lambda_eff)
        kin = self.kinetic_density(a, pi, K)
        E2 = self.p.Omega_m * a**-3 + self.p.Omega_r * a**-4 + V + kin
        E = np.sqrt(E2)
        chi_N = pi / (a**3 * K * E)
        w = (kin - V) / np.maximum(kin + V, 1.0e-30)
        F_bind = self.collapsed_fraction(a)
        dFbind_dN = self.dFbind_dN(a)
        dlnrhoB_dN = self.dlnrhoB_dN(a)
        dlnK_dlnB = self.dlnK_dlnB(a)
        kappa = self.kappa(a)
        dE2_dN = -3.0 * self.p.Omega_m * a**-3 - 4.0 * self.p.Omega_r * a**-4 - 6.0 * kin
        dlnE_dN = 0.5 * dE2_dN / E2
        dkappa_dN = np.gradient(np.asarray(kappa, dtype=np.float64), N, edge_order=2)

        out = {
            "lambda_eff": lambda_eff,
            "N": N,
            "a": a,
            "z": 1.0 / a - 1.0,
            "chi": chi,
            "pi": pi,
            "K": K,
            "rho_m": self.matter_density(a),
            "F_bind": F_bind,
            "dFbind_dN": dFbind_dN,
            "structure_boost": self.structure_boost(a),
            "rhoB": self.rhoB(a),
            "env_response": np.full_like(a, self.env_response()),
            "V": V,
            "kin": kin,
            "E": E,
            "E2": E2,
            "chi_N": chi_N,
            "w": w,
            "dlnE_dN": dlnE_dN,
            "dlnrhoB_dN": dlnrhoB_dN,
            "dlnK_dlnB": dlnK_dlnB,
            "kappa": kappa,
            "dkappa_dN": dkappa_dN,
            "dlnK_dN": self.dlnK_dN(a),
        }
        self._background = out
        return out

    def solve_growth_qs(self, k_hmpc: float = 0.1, samples: int = 1200):
        bg = self.solve_background(samples=max(samples, 1600))
        N = bg["N"]
        a = bg["a"]
        E = bg["E"]
        E2 = bg["E2"]
        chi = bg["chi"]
        chi_N = bg["chi_N"]
        K = bg["K"]
        dlnE_dN = bg["dlnE_dN"]
        dlnK_dN = bg["dlnK_dN"]
        kappa = bg["kappa"]
        dkappa_dN = bg["dkappa_dN"]
        dV = self.dV(chi)
        ddV = self.ddV(chi)

        z = 1.0 / a - 1.0
        k_mpc = k_hmpc * self.p.h
        kh_over_aH = k_mpc / np.maximum(a * self.p.H0_over_c_Mpc * E, 1.0e-30)
        k2_over_aH2 = kh_over_aH * kh_over_aH

        interp = {
            key: interp1d(N, value, kind="cubic", bounds_error=False, fill_value="extrapolate")
            for key, value in {
                "a": a,
                "E": E,
                "E2": E2,
                "K": K,
                "chi_N": chi_N,
                "dlnE_dN": dlnE_dN,
                "dlnK_dN": dlnK_dN,
                "kappa": kappa,
                "dkappa_dN": dkappa_dN,
                "dV": dV,
                "ddV": ddV,
                "k2_over_aH2": k2_over_aH2,
            }.items()
        }

        def rhs(Nloc: float, y: np.ndarray) -> np.ndarray:
            D, U, dchi, vchi = y
            aloc = float(interp["a"](Nloc))
            E2loc = float(interp["E2"](Nloc))
            Kloc = float(interp["K"](Nloc))
            dlnEloc = float(interp["dlnE_dN"](Nloc))
            dlnKloc = float(interp["dlnK_dN"](Nloc))
            kappaloc = float(interp["kappa"](Nloc))
            dkappa_dNloc = float(interp["dkappa_dN"](Nloc))
            dVloc = float(interp["dV"](Nloc))
            ddVloc = float(interp["ddV"](Nloc))
            chiNloc = float(interp["chi_N"](Nloc))
            k2loc = float(interp["k2_over_aH2"](Nloc))

            rho_m = self.p.Omega_m * aloc ** -3
            omega_m_a = rho_m / E2loc

            delta_rho_cdc = (
                Kloc * E2loc * chiNloc * vchi
                + dVloc * dchi
                + 0.5 * Kloc * E2loc * chiNloc * chiNloc * kappaloc * D
            )
            mu_eff = 1.0 + delta_rho_cdc / np.maximum(rho_m * D, 1.0e-30)

            kappa_N_delta = dkappa_dNloc * D + kappaloc * U
            source = -kappa_N_delta * chiNloc + kappaloc * D * dVloc / np.maximum(Kloc * E2loc, 1.0e-30)

            dD = U
            dU = -(2.0 + dlnEloc) * U + 1.5 * omega_m_a * mu_eff * D
            ddchi = vchi
            dvchi = (
                -(3.0 + dlnEloc + dlnKloc) * vchi
                - (k2loc / np.maximum(Kloc, 1.0e-30) + ddVloc / np.maximum(Kloc * E2loc, 1.0e-30)) * dchi
                + source
            )
            return np.array([dD, dU, ddchi, dvchi])

        N_ini = float(N[0])
        a_ini = float(a[0])
        y0 = np.array([a_ini, a_ini, 0.0, 0.0])
        sol = solve_ivp(
            rhs,
            (N_ini, 0.0),
            y0,
            method="BDF",
            dense_output=True,
            t_eval=N,
            rtol=1.0e-8,
            atol=1.0e-10,
            max_step=0.05,
        )
        if not sol.success:
            raise RuntimeError(f"CDC growth solve failed: {sol.message}")

        D = sol.y[0]
        U = sol.y[1]
        dchi = sol.y[2]
        vchi = sol.y[3]
        f = U / np.maximum(D, 1.0e-30)
        D_norm = D / D[-1]
        fsigma8 = self.p.sigma8_0 * D_norm * f

        rho_m = self.p.Omega_m * a**-3
        delta_rho_cdc = K * E2 * chi_N * vchi + dV * dchi + 0.5 * K * E2 * chi_N * chi_N * kappa * D
        mu_eff = 1.0 + delta_rho_cdc / np.maximum(rho_m * D, 1.0e-30)

        return {
            "N": N,
            "a": a,
            "z": z,
            "D": D_norm,
            "f": f,
            "fsigma8": fsigma8,
            "delta_chi": dchi,
            "delta_chi_N": vchi,
            "mu_eff": mu_eff,
            "eta": np.ones_like(mu_eff),
        }


def lcdm_E(z: np.ndarray | float, Omega_m: float, Omega_r: float) -> np.ndarray | float:
    Omega_de = 1.0 - Omega_m - Omega_r
    return np.sqrt(Omega_m * (1.0 + z) ** 3 + Omega_r * (1.0 + z) ** 4 + Omega_de)


def plot_reference_outputs(model: CDCReferenceModel, output_dir: Path, k_hmpc: float = 0.1):
    from cdc_upgrade.postprocess.simple_svg import write_three_panel_svg

    output_dir.mkdir(parents=True, exist_ok=True)
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
        output_dir / "cdc_background_growth_overview.svg",
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

    summary = {
        "params": asdict(model.p),
        "lambda_eff": float(bg["lambda_eff"]),
        "w0": float(bg["w"][-1]),
        "K0": float(bg["K"][-1]),
        "chi0": float(bg["chi"][-1]),
        "mu0_k0p1": float(gr["mu_eff"][-1]),
        "eta0_k0p1": float(gr["eta"][-1]),
        "fsigma8_0": float(gr["fsigma8"][-1]),
    }
    with open(output_dir / "cdc_reference_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def main():
    parser = argparse.ArgumentParser(description="CDC reference background and growth solver.")
    parser.add_argument("--output-dir", type=Path, default=Path("C:/Research/cdc_journal/cdc_upgrade/figures"))
    parser.add_argument("--k-hmpc", type=float, default=0.1)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    params = CDCParams()
    model = CDCReferenceModel(params)
    bg = model.solve_background()
    gr = model.solve_growth_qs(k_hmpc=args.k_hmpc)

    if args.smoke_test:
        print("CDC smoke test")
        print(f"lambda_eff = {bg['lambda_eff']:.8f}")
        print(f"w0         = {bg['w'][-1]:.6f}")
        print(f"K0         = {bg['K'][-1]:.6f}")
        print(f"mu0        = {gr['mu_eff'][-1]:.6f}")
        print(f"eta0       = {gr['eta'][-1]:.6f}")
        print(f"fsigma8(0) = {gr['fsigma8'][-1]:.6f}")

    plot_reference_outputs(model, args.output_dir, k_hmpc=args.k_hmpc)


if __name__ == "__main__":
    main()
