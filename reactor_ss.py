"""
reactor_ss.py -- Stage 1: steady-state, single-pass plug-flow reactor for CO2
hydrogenation to methanol over Cu/ZnO/Al2O3.

Independent variable is catalyst mass W [kg]. Two variants:

  integrate()               ISOTHERMAL. State = molar flows F_i [mol/s].
                            T imposed and constant.
  integrate_nonisothermal() NON-ISOTHERMAL. State = (F_i, T). Adds the energy
                            balance so the bed computes its own axial
                            temperature profile (adiabatic or cooled).

Mole balances along the bed (no pressure drop), order kinetics.SPECIES:
    dF_CO2/dW   = -r_MeOH - r_RWGS
    dF_H2/dW    = -3*r_MeOH - r_RWGS
    dF_CH3OH/dW = +r_MeOH
    dF_H2O/dW   = +r_MeOH + r_RWGS
    dF_CO/dW    = +r_RWGS

Energy balance (non-isothermal variant):
    dT/dW = [ (-dH1)*r_MeOH + (-dH2)*r_RWGS - Uc*(T - T_cool) ] / sum(F_i * cp_i)
  with Uc = U*a/rho_b [W/(kg_cat*K)] a lumped cooling coefficient (Uc=0 =>
  adiabatic). The exothermic methanol reaction drives the hotspot; the
  endothermic RWGS partly offsets it -- the reason CO2-fed hotspots are milder
  than syngas ones. This dT/dW term is exactly what gains a time derivative in
  Stage 2 and what sets the achievable ramp rate.

Still NOT included: pressure drop, recycle, separation, dynamics, ML.
"""

import numpy as np
from scipy.integrate import solve_ivp

import thermo
from kinetics import rates, SPECIES

# Heat-capacity vector in SPECIES order [J/mol/K], assembled from thermo.CP.
_CP_VEC = np.array([thermo.CP[s] for s in SPECIES])

# Small positive floor for product flows to keep the equilibrium terms and
# partial-pressure ratios well-defined at the inlet (seed at 1e-8, not 0).
_FLOOR = 1e-8


def _dFdW(W, F, T, P):
    Ft = F.sum()
    p = P * F / Ft                       # partial pressures [bar]
    r_meoh, r_rwgs = rates(p, T)
    return np.array([
        -r_meoh - r_rwgs,                # CO2
        -3.0 * r_meoh - r_rwgs,          # H2
        r_meoh,                          # CH3OH
        r_meoh + r_rwgs,                 # H2O
        r_rwgs,                          # CO
    ])


def integrate(F0, T, P, W_cat, rtol=1e-8, atol=1e-12, n_eval=200):
    """Integrate the isothermal PFR from W=0 to W=W_cat.

    Parameters
    ----------
    F0    : array-like, 5 inlet molar flows [mol/s], order = SPECIES.
    T, P  : temperature [K], total pressure [bar] (both constant).
    W_cat : total catalyst mass [kg].

    Returns
    -------
    dict with keys:
        W        : sampled catalyst-mass grid [kg]
        F        : (5, n_eval) molar-flow profiles [mol/s]
        F_out    : outlet molar flows [mol/s]
        X_CO2    : per-pass CO2 conversion [-]
        S_MeOH   : carbon selectivity to methanol, MeOH/(MeOH+CO) [-]
        Y_MeOH   : per-pass methanol yield on CO2 [-]
        sol      : raw solve_ivp result
    """
    F0 = np.array(F0, dtype=float)
    F0 = np.where(F0 <= 0.0, _FLOOR, F0)

    sol = solve_ivp(
        _dFdW, (0.0, W_cat), F0, args=(T, P),
        method="LSODA", rtol=rtol, atol=atol,
        t_eval=np.linspace(0.0, W_cat, n_eval),
    )
    if not sol.success:
        raise RuntimeError(f"PFR integration failed: {sol.message}")

    F_out = sol.y[:, -1]
    i = {s: k for k, s in enumerate(SPECIES)}
    X_CO2 = (F0[i["CO2"]] - F_out[i["CO2"]]) / F0[i["CO2"]]
    meoh, co = F_out[i["CH3OH"]], F_out[i["CO"]]
    S_MeOH = meoh / (meoh + co)
    Y_MeOH = meoh / F0[i["CO2"]]

    return {
        "W": sol.t, "F": sol.y, "F_out": F_out,
        "X_CO2": X_CO2, "S_MeOH": S_MeOH, "Y_MeOH": Y_MeOH, "sol": sol,
    }


# --------------------------------------------------------------------------- #
# Non-isothermal variant: 6-state (5 flows + temperature) energy balance.
# --------------------------------------------------------------------------- #
def _dYdW(W, Y, P, Uc, T_cool):
    F = Y[:5]
    T = Y[5]
    Ft = F.sum()
    p = P * F / Ft
    r_meoh, r_rwgs = rates(p, T)

    dF = np.array([
        -r_meoh - r_rwgs,                # CO2
        -3.0 * r_meoh - r_rwgs,          # H2
        r_meoh,                          # CH3OH
        r_meoh + r_rwgs,                 # H2O
        r_rwgs,                          # CO
    ])
    # Heat generated (J/kg_cat/s) minus heat removed by coolant, over the
    # heat-capacity flow (J/s/K) -> dT/dW [K/kg_cat].
    q_gen = (-thermo.DH_MEOH) * r_meoh + (-thermo.DH_RWGS) * r_rwgs
    q_cool = Uc * (T - T_cool)
    dT = (q_gen - q_cool) / np.dot(F, _CP_VEC)
    return np.concatenate([dF, [dT]])


def integrate_nonisothermal(F0, T_in, P, W_cat, mode="cooled",
                            T_cool=None, Uc=8.0,
                            rtol=1e-8, atol=1e-10, n_eval=400):
    """Integrate the non-isothermal PFR from W=0 to W=W_cat.

    Parameters
    ----------
    F0    : 5 inlet molar flows [mol/s], order = SPECIES.
    T_in  : inlet gas temperature [K].
    P     : total pressure [bar] (constant).
    W_cat : total catalyst mass [kg].
    mode  : 'cooled' (coolant present) or 'adiabatic' (no heat removal).
    T_cool: coolant temperature [K]; defaults to T_in. Ignored if adiabatic.
    Uc    : lumped cooling coefficient U*a/rho_b [W/(kg_cat*K)]. Set to 0 for
            adiabatic; 'adiabatic' mode forces this.

    Returns
    -------
    dict: as integrate(), plus
        T        : (n_eval,) axial temperature profile [K]
        T_out    : outlet temperature [K]
        T_max    : peak (hotspot) temperature [K]
        W_hot    : catalyst mass at the hotspot [kg]
        dT_hot   : hotspot rise above inlet, T_max - T_in [K]
    """
    if mode == "adiabatic":
        Uc = 0.0
    if T_cool is None:
        T_cool = T_in

    F0 = np.array(F0, dtype=float)
    F0 = np.where(F0 <= 0.0, _FLOOR, F0)
    Y0 = np.concatenate([F0, [float(T_in)]])

    sol = solve_ivp(
        _dYdW, (0.0, W_cat), Y0, args=(P, Uc, T_cool),
        method="LSODA", rtol=rtol, atol=atol,
        t_eval=np.linspace(0.0, W_cat, n_eval),
    )
    if not sol.success:
        raise RuntimeError(f"Non-isothermal integration failed: {sol.message}")

    F = sol.y[:5, :]
    T = sol.y[5, :]
    F_out = F[:, -1]
    i = {s: k for k, s in enumerate(SPECIES)}
    X_CO2 = (F0[i["CO2"]] - F_out[i["CO2"]]) / F0[i["CO2"]]
    meoh, co = F_out[i["CH3OH"]], F_out[i["CO"]]
    S_MeOH = meoh / (meoh + co)
    Y_MeOH = meoh / F0[i["CO2"]]
    j_hot = int(np.argmax(T))

    return {
        "W": sol.t, "F": F, "T": T, "F_out": F_out, "T_out": T[-1],
        "X_CO2": X_CO2, "S_MeOH": S_MeOH, "Y_MeOH": Y_MeOH,
        "T_max": T[j_hot], "W_hot": sol.t[j_hot], "dT_hot": T[j_hot] - T_in,
        "sol": sol,
    }


if __name__ == "__main__":
    # One green-methanol operating point: pure CO2 + H2 feed, H2/CO2 = 3.
    T, P = 503.0, 50.0
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])   # mol/s
    out = integrate(F0, T, P, W_cat=1.0)
    print(f"T = {T:.0f} K   P = {P:.0f} bar   H2/CO2 = 3")
    print(f"  CO2 conversion : {out['X_CO2']:.3f}")
    print(f"  MeOH selectivity: {out['S_MeOH']:.3f}")
    print(f"  MeOH yield      : {out['Y_MeOH']:.3f}")
