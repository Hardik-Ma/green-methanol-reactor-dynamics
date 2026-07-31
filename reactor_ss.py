"""
reactor_ss.py -- Stage 1: steady-state, single-pass, isothermal plug-flow
reactor for CO2 hydrogenation to methanol over Cu/ZnO/Al2O3.

Independent variable is catalyst mass W [kg]; state is the vector of molar
flows F_i [mol/s], ordered as kinetics.SPECIES = (CO2, H2, CH3OH, H2O, CO).

Mole balances along the bed (isothermal, no pressure drop):
    dF_CO2/dW   = -r_MeOH - r_RWGS
    dF_H2/dW    = -3*r_MeOH - r_RWGS
    dF_CH3OH/dW = +r_MeOH
    dF_H2O/dW   = +r_MeOH + r_RWGS
    dF_CO/dW    = +r_RWGS

Deliberately NOT included at this stage: energy balance, pressure drop,
recycle, separation, dynamics, ML. This is the verified kinetic kernel that
every later stage wraps.
"""

import numpy as np
from scipy.integrate import solve_ivp

from kinetics import rates, SPECIES

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


if __name__ == "__main__":
    # One green-methanol operating point: pure CO2 + H2 feed, H2/CO2 = 3.
    T, P = 503.0, 50.0
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])   # mol/s
    out = integrate(F0, T, P, W_cat=1.0)
    print(f"T = {T:.0f} K   P = {P:.0f} bar   H2/CO2 = 3")
    print(f"  CO2 conversion : {out['X_CO2']:.3f}")
    print(f"  MeOH selectivity: {out['S_MeOH']:.3f}")
    print(f"  MeOH yield      : {out['Y_MeOH']:.3f}")
