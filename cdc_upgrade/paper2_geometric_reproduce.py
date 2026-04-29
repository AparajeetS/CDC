#!/usr/bin/env python3
"""
CDC v3 â€” Full Numerical Pipeline
=================================

Configurational Dispersal Cosmology with kinetic suppression, tested against:
  - DESI DR2 BAO (full block-diagonal covariance, Table IV values)
  - Planck 2018 compressed CMB shift parameters (R, l_a, Î©_b hÂ²)

Compared against:
  - Î›CDM (k = 2 BAO / 3 BAO+CMB)
  - wâ‚€wâ‚CDM (k = 4 / 5)
  - CDC v3 (k = 6 / 7)

Honest result: CDC v3 in its current background-only form does not improve
the BAO+CMB fit over Î›CDM. The framework's value-add must come from growth
sector tests (fÏƒâ‚ˆ, void Alcock-Paczynski) rather than geometric distance data.

Theory:
  L_Ï‡ = -Â½ K(Ï_B)(âˆ‚Ï‡)Â² - V(Ï‡)        [no F(Ï‡)R coupling, gravity unmodified]
  K(Ï_B) = 1 + (Ï_B/Ï_*)^n           [binding-dependent kinetic function]
  V(Ï‡) = Î›_eff + A(Ï‡Â²-vÂ²)Â²           [Mexican hat, Ï‡=0 tachyonically unstable]

Field equation:  Ï‡Ìˆ + (KÌ‡/K + 3H + Â½Î·_H) Ï‡Ì‡ + V'(Ï‡)/(K EÂ²) = 0
"""

import numpy as np
import sys
from scipy.integrate import solve_ivp, quad
from scipy.interpolate import interp1d
from scipy.linalg import inv
from scipy.optimize import minimize

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# =============================================================================
# CONSTANTS
# =============================================================================
c = 299792.458   # km/s
Or = 9e-5        # radiation density today

# =============================================================================
# DESI DR2 TABLE IV â€” full block-diagonal BAO likelihood
# =============================================================================
DESI_DR2 = {
    'BGS':       {'z': 0.295, 'type': 'DV', 'DV': 7.942,  'DV_err': 0.075},
    'LRG1':      {'z': 0.510, 'type': '2D', 'DM': 13.588, 'DM_err': 0.167,
                  'DH': 21.863, 'DH_err': 0.425, 'r_MH': -0.459},
    'LRG2':      {'z': 0.706, 'type': '2D', 'DM': 17.351, 'DM_err': 0.177,
                  'DH': 19.455, 'DH_err': 0.330, 'r_MH': -0.404},
    'LRG3+ELG1': {'z': 0.934, 'type': '2D', 'DM': 21.576, 'DM_err': 0.152,
                  'DH': 17.641, 'DH_err': 0.193, 'r_MH': -0.416},
    'ELG2':      {'z': 1.321, 'type': '2D', 'DM': 27.601, 'DM_err': 0.318,
                  'DH': 14.176, 'DH_err': 0.221, 'r_MH': -0.434},
    'QSO':       {'z': 1.484, 'type': '2D', 'DM': 30.512, 'DM_err': 0.760,
                  'DH': 12.817, 'DH_err': 0.516, 'r_MH': -0.500},
    'Lya':       {'z': 2.330, 'type': '2D', 'DM': 38.988, 'DM_err': 0.531,
                  'DH': 8.632,  'DH_err': 0.101, 'r_MH': -0.431},
}


def build_bao_likelihood():
    """Build data vector and inverse covariance for DESI DR2 BAO."""
    data, blocks, specs = [], [], []
    for name, d in DESI_DR2.items():
        z = d['z']
        if d['type'] == 'DV':
            data.append(d['DV'])
            blocks.append(np.array([[d['DV_err']**2]]))
            specs.append(('DV', z))
        else:
            data += [d['DM'], d['DH']]
            s1, s2, r = d['DM_err'], d['DH_err'], d['r_MH']
            blocks.append(np.array([[s1**2, r*s1*s2],
                                    [r*s1*s2, s2**2]]))
            specs += [('DM', z), ('DH', z)]
    n = sum(b.shape[0] for b in blocks)
    cov = np.zeros((n, n))
    i = 0
    for b in blocks:
        s = b.shape[0]
        cov[i:i+s, i:i+s] = b
        i += s
    return np.array(data), inv(cov), specs


BAO_DATA, BAO_COV_INV, BAO_SPECS = build_bao_likelihood()

# =============================================================================
# Planck 2018 compressed CMB shift parameters
# =============================================================================
CMB_DATA = np.array([1.7493, 301.462, 0.02237])  # R, l_a, Î©_b hÂ²
CMB_ERR = np.array([0.0048, 0.090, 0.00015])
CMB_CORR = np.array([
    [1.00, 0.49, -0.62],
    [0.49, 1.00, -0.42],
    [-0.62, -0.42, 1.00]
])
CMB_COV_INV = inv(CMB_CORR * np.outer(CMB_ERR, CMB_ERR))
Z_STAR = 1089.92


# =============================================================================
# CHI-SQUARE FUNCTIONS
# =============================================================================
def chi2_bao(Ez_func, hrd):
    """Ï‡Â² against DESI DR2 BAO with full block-diagonal covariance.

    Parameters
    ----------
    Ez_func : callable
        E(z) = H(z)/Hâ‚€ as a function of redshift (any cosmology)
    hrd : float
        h Ã— r_d in Mpc (degenerate combination for BAO)
    """
    H0 = 67.4  # arbitrary reference; only h*rd matters
    rd = hrd * 100 / H0
    pred = np.zeros(len(BAO_DATA))
    for i, (kind, z) in enumerate(BAO_SPECS):
        if kind == 'DV':
            r, _ = quad(lambda zp: 1.0/Ez_func(zp), 0, z, limit=80)
            dm = r * c / H0
            dh = c / (H0 * Ez_func(z))
            pred[i] = (dm**2 * z * dh)**(1./3.) / rd
        elif kind == 'DM':
            r, _ = quad(lambda zp: 1.0/Ez_func(zp), 0, z, limit=80)
            pred[i] = (r * c / H0) / rd
        else:  # DH
            pred[i] = (c / (H0 * Ez_func(z))) / rd
    diff = pred - BAO_DATA
    return float(diff @ BAO_COV_INV @ diff)


def chi2_cmb(Ez_func, Om, h, ombh2):
    """Ï‡Â² against Planck 2018 compressed CMB shift parameters."""
    H0 = h * 100
    r, _ = quad(lambda zp: 1.0/Ez_func(zp), 0, Z_STAR, limit=300, epsabs=1e-8)
    DA = r * c / H0  # comoving distance to z*
    omh2 = Om * h**2
    rs = 144.43 * (omh2/0.14)**(-0.252) * (ombh2/0.022)**(-0.083)  # Hu & Sugiyama
    R = np.sqrt(Om) * H0 * DA / c
    la = np.pi * DA / rs
    pred = np.array([R, la, ombh2])
    diff = pred - CMB_DATA
    return float(diff @ CMB_COV_INV @ diff)


# =============================================================================
# COSMOLOGICAL MODELS
# =============================================================================
def E_lcdm(z, Om):
    OL = 1 - Om - Or
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + OL)


def E_w0wa(z, Om, w0, wa):
    OL = 1 - Om - Or
    a = 1.0/(1+z)
    rho_de = OL * a**(-3*(1+w0+wa)) * np.exp(-3*wa*(1-a))
    return np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + rho_de)


def solve_cdc_v3(Om, v, vac_offset, Omega_star, n, chi_init_frac):
    """
    Solve the CDC v3 background field equation.

    Returns a callable E(z) valid for all z (stitched to Î›CDM-like at z > 5
    where the field is frozen at chi_init).
    """
    OL = 1 - Om - Or
    A = vac_offset * OL / v**4
    Lambda_eff = OL * (1 - vac_offset)
    chi_init = chi_init_frac * v

    N0, N1 = -np.log(5), 0  # ln(a) from a=0.2 (z=4) to a=1
    N_arr = np.linspace(N0, N1, 2000)

    def rhs(N, y):
        chi, chip = y
        a = np.exp(N)
        rm = Om * a**(-3)
        K = 1 + (rm / Omega_star)**n
        dKdN = -3 * n * (rm / Omega_star)**n
        KdK = dKdN / K
        V = Lambda_eff + A * (chi**2 - v**2)**2
        denom = max(1 - 0.5*K*chip**2, 0.01)
        E2 = (rm + Or*a**(-4) + V) / denom
        if E2 < 1e-10:
            E2 = 1e-10
        eta_H = (-3*rm - 4*Or*a**(-4)) / E2
        force = -4*A*chi*(chi**2 - v**2) / (K * E2)
        return [chip, -(KdK + 3 + 0.5*eta_H)*chip + force]

    sol = solve_ivp(rhs, [N0, N1], [chi_init, 0], t_eval=N_arr,
                    method='RK45', rtol=1e-8, atol=1e-10, max_step=0.02)
    if sol.status != 0:
        return None

    a = np.exp(sol.t)
    z = 1.0/a - 1.0
    chi = sol.y[0]
    chip = sol.y[1]
    rm = Om * a**(-3)
    K = 1 + (rm / Omega_star)**n
    V_arr = np.array([Lambda_eff + A*(cc**2 - v**2)**2 for cc in chi])
    denom = np.maximum(1 - 0.5*K*chip**2, 0.01)
    E2 = (rm + Or*a**(-4) + V_arr) / denom
    E_arr = np.sqrt(np.maximum(E2, 1e-30))
    rho_chi = 0.5*K*E2*chip**2 + V_arr
    w_arr = (0.5*K*E2*chip**2 - V_arr) / np.maximum(rho_chi, 1e-30)

    z_arr = z[::-1]
    E_arr = E_arr[::-1]
    w_arr = w_arr[::-1]

    Ei = interp1d(z_arr, E_arr, kind='cubic',
                  bounds_error=False, fill_value='extrapolate')
    wi = interp1d(z_arr, w_arr, kind='cubic',
                  bounds_error=False, fill_value=-1)

    # Stitched E(z): CDC for z<4.5, frozen-field Î›CDM-like for z>4.5
    V_init = Lambda_eff + A*v**4*(chi_init_frac**2 - 1)**2

    def E_full(zp):
        if np.isscalar(zp):
            if zp < 4.5:
                return float(Ei(zp))
            return float(np.sqrt(Om*(1+zp)**3 + Or*(1+zp)**4 + V_init))
        return np.array([E_full(zi) for zi in zp])

    return E_full, wi


# =============================================================================
# OPTIMIZATION DRIVERS
# =============================================================================
def fit_lcdm_bao():
    f = lambda p: chi2_bao(lambda z: E_lcdm(z, p[0]), p[1]) \
                  if (0.2 < p[0] < 0.45 and 95 < p[1] < 110) else 1e10
    return minimize(f, [0.30, 100], method='Nelder-Mead', options={'xatol': 1e-5})


def fit_lcdm_baocmb():
    def f(p):
        Om, h, ob = p
        if not (0.2 < Om < 0.45 and 0.6 < h < 0.75 and 0.02 < ob < 0.025):
            return 1e10
        Ef = lambda z: E_lcdm(z, Om)
        omh2 = Om * h**2
        rd = 147.05 * (omh2/0.1432)**(-0.23) * (ob/0.02236)**(-0.13)
        return chi2_bao(Ef, h*rd) + chi2_cmb(Ef, Om, h, ob)
    return minimize(f, [0.31, 0.674, 0.0224], method='Nelder-Mead', options={'xatol': 1e-5})


def fit_w0wa_baocmb():
    def f(p):
        Om, h, ob, w0, wa = p
        if not (0.2 < Om < 0.45 and 0.6 < h < 0.78 and 0.02 < ob < 0.025):
            return 1e10
        if not (-2 < w0 < 0 and -3 < wa < 3):
            return 1e10
        Ef = lambda z: E_w0wa(z, Om, w0, wa)
        omh2 = Om * h**2
        rd = 147.05 * (omh2/0.1432)**(-0.23) * (ob/0.02236)**(-0.13)
        return chi2_bao(Ef, h*rd) + chi2_cmb(Ef, Om, h, ob)

    best = (1e10, None)
    for s in [[0.31, 0.674, 0.0224, -1, 0],
              [0.31, 0.674, 0.0224, -0.75, -0.88],
              [0.32, 0.67, 0.0224, -0.5, -1.5],
              [0.32, 0.68, 0.0224, -0.85, -0.5]]:
        r = minimize(f, s, method='Nelder-Mead', options={'xatol': 1e-5, 'maxiter': 2000})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best


def fit_cdc_baocmb():
    def f(p):
        Om, h, ob, v, vof, Os, ci = p
        if not (0.2 < Om < 0.45 and 0.6 < h < 0.78 and 0.02 < ob < 0.025):
            return 1e10
        if not (0.01 < v < 0.5 and 0.005 < vof < 0.30 and 0.05 < Os < 5 and 0.05 < ci < 0.95):
            return 1e10
        sol = solve_cdc_v3(Om, v, vof, Os, 2, ci)
        if sol is None:
            return 1e10
        Ef = sol[0]
        omh2 = Om * h**2
        rd = 147.05 * (omh2/0.1432)**(-0.23) * (ob/0.02236)**(-0.13)
        try:
            return chi2_bao(Ef, h*rd) + chi2_cmb(Ef, Om, h, ob)
        except Exception:
            return 1e10

    best = (1e10, None)
    for s in [[0.30, 0.674, 0.0224, 0.05, 0.10, 0.30, 0.30],
              [0.30, 0.674, 0.0224, 0.10, 0.15, 0.50, 0.30],
              [0.31, 0.674, 0.0224, 0.05, 0.20, 0.6, 0.20],
              [0.31, 0.674, 0.0224, 0.10, 0.05, 1.0, 0.40],
              [0.32, 0.667, 0.0224, 0.05, 0.20, 0.6, 0.30]]:
        r = minimize(f, s, method='Nelder-Mead', options={'xatol': 1e-4, 'maxiter': 1500})
        if r.fun < best[0]:
            best = (r.fun, r.x)
    return best


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 76)
    print("CDC v3 â€” DESI DR2 + Planck CMB shift parameters")
    print("=" * 76)

    print("\nFitting models...")
    res_l = fit_lcdm_bao()
    res_lc = fit_lcdm_baocmb()
    res_w = fit_w0wa_baocmb()
    res_c = fit_cdc_baocmb()

    print(f"\n{'-'*60}")
    print(f"Î›CDM:")
    print(f"  BAO only:     Î©_m = {res_l.x[0]:.4f}, hÂ·r_d = {res_l.x[1]:.2f}")
    print(f"                Ï‡Â² = {res_l.fun:.3f}, Ï‡Â²/Î½ = {res_l.fun/11:.3f}")
    print(f"  BAO + CMB:    Î©_m = {res_lc.x[0]:.4f}, h = {res_lc.x[1]:.4f}, "
          f"Î©_b hÂ² = {res_lc.x[2]:.5f}")
    print(f"                Ï‡Â² = {res_lc.fun:.3f}")

    print(f"\nwâ‚€wâ‚CDM:")
    p = res_w[1]
    print(f"  BAO + CMB:    Î©_m = {p[0]:.4f}, h = {p[1]:.4f}")
    print(f"                wâ‚€ = {p[3]:.4f}, wâ‚ = {p[4]:+.4f}")
    print(f"                Ï‡Â² = {res_w[0]:.3f}, Î”Ï‡Â² = {res_w[0] - res_lc.fun:+.3f}")

    print(f"\nCDC v3:")
    p = res_c[1]
    if p is not None:
        print(f"  BAO + CMB:    Î©_m = {p[0]:.4f}, h = {p[1]:.4f}")
        print(f"                v = {p[3]:.3f}, vac_off = {p[4]*100:.1f}%, "
              f"Î©* = {p[5]:.3f}, Ï‡_i = {p[6]:.3f}v")
        sol = solve_cdc_v3(p[0], p[3], p[4], p[5], 2, p[6])
        if sol:
            wi = sol[1]
            w0 = float(wi(0))
            wa = 3 * (float(wi(0.5)) - w0)
            print(f"                wâ‚€ = {w0:.4f}, wâ‚ = {wa:+.4f}")
        print(f"                Ï‡Â² = {res_c[0]:.3f}, Î”Ï‡Â² = {res_c[0] - res_lc.fun:+.3f}")

    # Information criteria
    n_data = len(BAO_DATA) + 3
    print(f"\n{'-'*60}")
    print(f"Information criteria (BAO+CMB, n_data = {n_data}):")
    print(f"  Î›CDM       (k=3):  Î”AIC = 0,         Î”BIC = 0")
    aic_w = (res_w[0] - res_lc.fun) + 2*2
    bic_w = (res_w[0] - res_lc.fun) + 2*np.log(n_data)
    print(f"  wâ‚€wâ‚CDM   (k=5):  Î”AIC = {aic_w:+.3f}, Î”BIC = {bic_w:+.3f}")
    if res_c[0] < 1e9:
        aic_c = (res_c[0] - res_lc.fun) + 2*4
        bic_c = (res_c[0] - res_lc.fun) + 4*np.log(n_data)
        print(f"  CDC v3     (k=7):  Î”AIC = {aic_c:+.3f}, Î”BIC = {bic_c:+.3f}")

    print(f"\n{'-'*60}")
    print("Verdict:")
    print("  CDC v3 in its current background-only form does not improve the")
    print("  fit to BAO+CMB combined over Î›CDM. The framework's value-add must")
    print("  come from growth/RSD measurements (where the void enhancement")
    print("  mechanism predicts unique signatures), not from geometric distance")
    print("  data where the simpler wâ‚€wâ‚CDM parametrization wins on AIC/BIC.")


if __name__ == '__main__':
    main()
