"""Cobaya CAMB wrapper for a structure-aware CDC background approximation.

This module maps the CDC parameter set

    {H0, ombh2, omch2, cdc_v, cdc_f_vac, cdc_Omega_star, cdc_chi_ratio,
     cdc_bind_amp_bg, cdc_bind_amp_env, cdc_bind_zhalf}

to a tabulated dark-energy equation of state w(a) computed by the stable
reference CDC background solver. CAMB then evolves the CMB and matter
perturbations using its standard DarkEnergyFluid treatment with cs2 = 1.

This is not the fully patched CDC perturbation theory. It is a full-likelihood
background-consistent approximation suitable for immediate Planck+DESI+Pantheon
sampling while the full custom Boltzmann backend is still under development.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cobaya.log import LoggedError
from cobaya.theories.camb.camb import CAMB

from cdc_upgrade.python.cdc_boltzmann_reference import CDCParams, CDCReferenceModel


class CDCCAMBApprox(CAMB):
    """CAMB-based CDC approximation using a tabulated non-phantom w(a)."""

    cdc_n: float = 2.0
    cdc_beta_B: float = 1.0
    cdc_bind_slope: float = 4.0
    cdc_env_bias: float = 1.0
    cdc_a_min: float = 1.0e-4
    cdc_table_size: int = 500

    def initialize(self):
        super().initialize()
        self.extra_args.setdefault("dark_energy_model", self.camb.dark_energy.DarkEnergyPPF)
        self.extra_args.setdefault("cs2", 1.0)

    def initialize_with_params(self):
        super().initialize_with_params()
        self._input_params_extra.update({"H0", "ombh2", "omch2", "mnu"})

    def get_helper_theories(self):
        helpers = super().get_helper_theories()
        helper = helpers.get("camb.transfers")
        if helper is not None:
            helper.input_params_extra.update(
                {
                    "cdc_v",
                    "cdc_f_vac",
                    "cdc_Omega_star",
                    "cdc_chi_ratio",
                    "cdc_chi_ini",
                    "cdc_bind_amp_bg",
                    "cdc_bind_amp_env",
                    "cdc_bind_zhalf",
                    "mnu",
                }
            )
        return helpers

    def get_can_support_params(self):
        return set(super().get_can_support_params()).union(
            {
                "cdc_v",
                "cdc_f_vac",
                "cdc_Omega_star",
                "cdc_chi_ratio",
                "cdc_chi_ini",
                "cdc_bind_amp_bg",
                "cdc_bind_amp_env",
                "cdc_bind_zhalf",
            }
        )

    def get_can_provide_params(self):
        return set(super().get_can_provide_params()).union(
            {"Omega_m", "cdc_w0", "cdc_wa", "cdc_lambda_eff", "cdc_chi0", "cdc_K0"}
        )

    @staticmethod
    def _omega_nu_approx(mnu: float, h: float) -> float:
        if h <= 0:
            return 0.0
        return mnu / (93.14 * h * h)

    def _build_cdc_reference(self, params_values_dict):
        H0 = float(params_values_dict["H0"])
        h = H0 / 100.0
        ombh2 = float(params_values_dict["ombh2"])
        omch2 = float(params_values_dict["omch2"])
        mnu = float(params_values_dict.get("mnu", 0.06))
        omega_nu = self._omega_nu_approx(mnu, h)
        Omega_m = (ombh2 + omch2 + omega_nu) / (h * h)
        Omega_b = ombh2 / (h * h)

        cdc_v = float(params_values_dict["cdc_v"])
        cdc_f_vac = float(params_values_dict["cdc_f_vac"])
        cdc_Omega_star = float(params_values_dict["cdc_Omega_star"])
        cdc_chi_ini = float(
            params_values_dict.get("cdc_chi_ini", params_values_dict["cdc_chi_ratio"] * cdc_v)
        )
        cdc_bind_amp_bg = float(params_values_dict.get("cdc_bind_amp_bg", params_values_dict.get("cdc_bind_amp", 0.0)))
        cdc_bind_amp_env = float(params_values_dict.get("cdc_bind_amp_env", 1.0))
        cdc_bind_zhalf = float(params_values_dict.get("cdc_bind_zhalf", 1.0))

        ref = CDCReferenceModel(
            CDCParams(
                Omega_m=Omega_m,
                Omega_b=Omega_b,
                h=h,
                v=cdc_v,
                f_vac=cdc_f_vac,
                Omega_star=cdc_Omega_star,
                chi_init=cdc_chi_ini,
                n=self.cdc_n,
                beta_B=self.cdc_beta_B,
                bind_amp_bg=cdc_bind_amp_bg,
                bind_amp_env=cdc_bind_amp_env,
                bind_zhalf=cdc_bind_zhalf,
                bind_slope=self.cdc_bind_slope,
                env_bias=self.cdc_env_bias,
                a_ini=self.cdc_a_min,
                a_eval_min=self.cdc_a_min,
                z_max_output=max(5.0, self.extra_args.get("zmax", 5.0)),
            )
        )
        bg = ref.solve_background(samples=max(int(self.cdc_table_size), 300))

        a = np.asarray(bg["a"], dtype=np.float64)
        w = np.asarray(bg["w"], dtype=np.float64)

        # CAMB requires ascending a and the table must end exactly at a=1.
        if not np.isclose(a[-1], 1.0):
            raise LoggedError(self.log, "CDC w(a) table does not end at a=1.")

        # Estimate CPL-like wa = -dw/da at a=1 for reporting only.
        dw_da_0 = np.gradient(w, a)[-1]
        summary = {
            "a": a,
            "w": w,
            "w0": float(w[-1]),
            "wa": float(-dw_da_0),
            "lambda_eff": float(bg["lambda_eff"]),
            "chi0": float(bg["chi"][-1]),
            "K0": float(bg["K"][-1]),
        }
        return summary

    def set(self, params_values_dict, state):
        try:
            cdc = self._build_cdc_reference(params_values_dict)
        except Exception as exc:
            if self.stop_at_error:
                raise
            self.log.debug("CDC reference solve failed: %s", exc)
            raise self.camb.baseconfig.CAMBError(f"CDC reference solve failed: {exc}") from exc

        args = dict(params_values_dict)
        for key in [
            "cdc_v",
            "cdc_f_vac",
            "cdc_Omega_star",
            "cdc_chi_ratio",
            "cdc_chi_ini",
            "cdc_bind_amp_bg",
            "cdc_bind_amp_env",
            "cdc_bind_zhalf",
        ]:
            args.pop(key, None)
        args["use_tabulated_w"] = True
        args["wde_a_array"] = cdc["a"]
        args["wde_w_array"] = cdc["w"]
        state["cdc_reference"] = cdc
        camb_params = super().set(args, state)
        if camb_params:
            setattr(camb_params, "_cdc_reference", cdc)
        return camb_params

    def _get_derived(self, p, intermediates):
        if p == "Omega_m":
            H0 = float(getattr(intermediates.camb_params, "H0", 0.0))
            h = H0 / 100.0 if H0 else 0.0
            if h:
                ombh2 = float(getattr(intermediates.camb_params, "ombh2", 0.0))
                omch2 = float(getattr(intermediates.camb_params, "omch2", 0.0))
                omnuh2 = float(getattr(intermediates.camb_params, "omnuh2", 0.0))
                return (ombh2 + omch2 + omnuh2) / (h * h)
            return None

        cdc = getattr(intermediates.camb_params, "_cdc_reference", {}) or {}
        cdc_map = {
            "cdc_w0": "w0",
            "cdc_wa": "wa",
            "cdc_lambda_eff": "lambda_eff",
            "cdc_chi0": "chi0",
            "cdc_K0": "K0",
        }
        if p in cdc_map:
            return cdc.get(cdc_map[p])

        return super()._get_derived(p, intermediates)

    def calculate(self, state, want_derived=True, **params_values_dict):
        return super().calculate(state, want_derived=want_derived, **params_values_dict)
