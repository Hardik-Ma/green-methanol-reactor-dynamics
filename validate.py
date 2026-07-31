"""
validate.py -- Stage 1 validation of the steady-state single-pass reactor.

Two checks, both self-contained (no external data needed):

1. EQUILIBRIUM ASYMPTOTE. The PFR, given a large catalyst mass, must converge
   to the composition predicted by an INDEPENDENT two-reaction chemical
   equilibrium solve built only from K_eq1, K_eq2. Agreement validates that the
   kinetics, thermodynamics, and mole balances are mutually consistent -- this
   is the rigorous "definition of done" for Stage 1, and it does not depend on
   any single literature operating point.

2. TEMPERATURE SWEEP (the first shippable figure). At a realistic catalyst
   load, CO2 conversion, methanol selectivity, and methanol yield vs T. Total
   CO2 conversion rises with T (endothermic RWGS keeps making CO), while
   methanol yield rises then rolls over as the exothermic methanol equilibrium
   and RWGS fight back -- the physically correct signature.

Run:  python -m src.plant.validate
"""

import os
import warnings
import numpy as np
from scipy.optimize import fsolve

from reactor_ss import integrate, _FLOOR
import thermo


# --------------------------------------------------------------------------- #
# Independent two-reaction equilibrium (methanol + RWGS), feed CO2 + H2 only.
# Solve the two equilibrium relations simultaneously for the reaction extents
# x1 (methanol) and x2 (RWGS). Softplus-style guarding keeps all flows > 0
# during the Newton iterations.
# --------------------------------------------------------------------------- #
def equilibrium(T, P, n_CO2=0.25, n_H2=0.75):
    """Return (X_CO2, S_MeOH) at chemical equilibrium. Pressures in bar."""
    K1, K2 = thermo.K_eq1(T), thermo.K_eq2(T)
    eps = 1e-12

    def resid(z):
        x1, x2 = z
        FCO2 = max(n_CO2 - x1 - x2, eps)
        FH2 = max(n_H2 - 3 * x1 - x2, eps)
        FMeOH = max(x1, eps)
        FH2O = max(x1 + x2, eps)
        FCO = max(x2, eps)
        Ft = n_CO2 + n_H2 - 2 * x1
        xr = lambda F: F / Ft
        # K1 [bar^-2] = xratio * P^(dn), dn = -2  ->  xratio = K1 * P^2
        e1 = (xr(FMeOH) * xr(FH2O)) / (xr(FCO2) * xr(FH2) ** 3) - K1 * P ** 2
        e2 = (FCO * FH2O) / (FCO2 * FH2) - K2   # RWGS, dn = 0
        return [e1, e2]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        x1, x2 = fsolve(resid, [0.03, 0.02], full_output=False)
        # No CO in the feed => net RWGS extent cannot be negative at
        # equilibrium. If the unconstrained solve pushes x2 < 0, the true
        # (constrained) state sits on the x2 = 0 boundary: solve methanol
        # equilibrium alone.
        if x2 < 0:
            x2 = 0.0
            x1 = fsolve(lambda x: resid([x[0], 0.0])[0], [0.03])[0]
    X_CO2 = (x1 + x2) / n_CO2
    S_MeOH = x1 / (x1 + x2)
    return X_CO2, S_MeOH


def check_equilibrium(conditions, W_big=500.0):
    print("=" * 70)
    print("CHECK 1 -- reactor (large W) vs independent equilibrium")
    print("=" * 70)
    print(f"{'T[K]':>6}{'P[bar]':>8}"
          f"{'X_eq':>9}{'X_rxr':>9}{'S_eq':>9}{'S_rxr':>9}")
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])
    for T, P in conditions:
        Xe, Se = equilibrium(T, P)
        o = integrate(F0, float(T), float(P), W_big)
        print(f"{T:>6.0f}{P:>8.0f}"
              f"{Xe:>9.3f}{o['X_CO2']:>9.3f}{Se:>9.3f}{o['S_MeOH']:>9.3f}")
    print()


def temperature_sweep(P=50.0, W_cat=30.0, T_grid=None, save_path=None):
    """CO2 conversion / MeOH selectivity / MeOH yield vs T. Returns arrays."""
    if T_grid is None:
        T_grid = np.linspace(453.0, 553.0, 26)
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])
    X, S, Y = [], [], []
    for T in T_grid:
        o = integrate(F0, float(T), float(P), W_cat)
        X.append(o["X_CO2"]); S.append(o["S_MeOH"]); Y.append(o["Y_MeOH"])
    X, S, Y = map(np.asarray, (X, S, Y))

    print("=" * 70)
    print(f"CHECK 2 -- temperature sweep  (P = {P:.0f} bar, W = {W_cat:.0f} kg)")
    print("=" * 70)
    print(f"{'T[K]':>6}{'X_CO2':>9}{'S_MeOH':>9}{'Y_MeOH':>9}")
    for T, x, s, y in zip(T_grid, X, S, Y):
        print(f"{T:>6.0f}{x:>9.3f}{s:>9.3f}{y:>9.3f}")
    print(f"\nMethanol yield peaks near T = {T_grid[int(np.argmax(Y))]:.0f} K")

    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Tc = T_grid - 273.15
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(Tc, 100 * X, "o-", label="CO$_2$ conversion", color="#1f4e79")
        ax.plot(Tc, 100 * S, "s-", label="MeOH selectivity", color="#c55a11")
        ax.plot(Tc, 100 * Y, "^-", label="MeOH yield", color="#2e7d32")
        ax.set_xlabel("Temperature (\u00b0C)")
        ax.set_ylabel("Percent (%)")
        ax.set_title(f"Single-pass CO$_2$ hydrogenation "
                     f"(P = {P:.0f} bar, H$_2$/CO$_2$ = 3, W = {W_cat:.0f} kg)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"\nFigure written to {save_path}")

    return T_grid, X, S, Y


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    results = os.path.abspath(os.path.join(here, "..", "..", "results"))
    os.makedirs(results, exist_ok=True)

    # Base green-methanol operating point at a realistic catalyst load.
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])
    base = integrate(F0, 503.0, 50.0, W_cat=30.0)
    print(f"Base case  T=503 K  P=50 bar  H2/CO2=3  W=30 kg")
    print(f"  X_CO2 = {base['X_CO2']:.3f}   S_MeOH = {base['S_MeOH']:.3f}"
          f"   Y_MeOH = {base['Y_MeOH']:.3f}\n")

    check_equilibrium([(483, 50), (503, 50), (523, 50), (503, 80)])
    temperature_sweep(P=50.0, W_cat=30.0,
                      save_path=os.path.join(results, "01_temperature_sweep.png"))
