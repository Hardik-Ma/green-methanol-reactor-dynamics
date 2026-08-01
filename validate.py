"""
validate.py -- Stage 1 validation of the steady-state single-pass reactor.

Checks, all self-contained (no external data needed):

1. EQUILIBRIUM ASYMPTOTE. The isothermal PFR, given a large catalyst mass, must
   converge to the composition predicted by an INDEPENDENT two-reaction
   chemical equilibrium solve built only from K_eq1, K_eq2. Agreement validates
   that the kinetics, thermodynamics, and mole balances are mutually consistent.

2. TEMPERATURE SWEEP (first shippable figure). CO2 conversion, methanol
   selectivity, and methanol yield vs T; total conversion rises with T while
   methanol yield rolls over -- the physically correct signature.

3. NON-ISOTHERMAL LIMIT CHECK. With strong cooling (Uc -> large), the energy
   balance must pin the bed at the coolant temperature and reproduce the
   isothermal result. This validates the new energy balance against Check 1.

4. AXIAL PROFILE + HOTSPOT (figure). Temperature down the bed for adiabatic vs
   cooled operation, showing the interior hotspot the coolant sets.

5. HOTSPOT VS LOAD (figure). Feed-proportional turndown: how peak temperature,
   hotspot location, and methanol yield move as the plant is turned down -- the
   first glimpse of flexible-operation behavior, still fully steady-state.
"""

import os
import warnings
import numpy as np
from scipy.optimize import fsolve

from reactor_ss import integrate, integrate_nonisothermal, _FLOOR
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


# --------------------------------------------------------------------------- #
# Check 3 -- non-isothermal energy balance recovers the isothermal limit.
# --------------------------------------------------------------------------- #
def check_nonisothermal_limit(T=503.0, P=50.0, W_cat=30.0):
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])
    iso = integrate(F0, T, P, W_cat)
    strong = integrate_nonisothermal(F0, T, P, W_cat, mode="cooled",
                                      T_cool=T, Uc=1e4)
    print("=" * 70)
    print("CHECK 3 -- non-isothermal limit: strong cooling -> isothermal")
    print("=" * 70)
    print(f"{'':>22}{'X_CO2':>9}{'S_MeOH':>9}{'Y_MeOH':>9}{'T_out[K]':>10}")
    print(f"{'isothermal':>22}{iso['X_CO2']:>9.3f}{iso['S_MeOH']:>9.3f}"
          f"{iso['Y_MeOH']:>9.3f}{T:>10.2f}")
    print(f"{'cooled (Uc=1e4)':>22}{strong['X_CO2']:>9.3f}{strong['S_MeOH']:>9.3f}"
          f"{strong['Y_MeOH']:>9.3f}{strong['T_out']:>10.2f}")
    print()


# --------------------------------------------------------------------------- #
# Check 4 -- axial temperature profile and hotspot (adiabatic vs cooled).
# --------------------------------------------------------------------------- #
def axial_profile(T_in=503.0, P=50.0, W_cat=30.0, T_cool=503.0, Uc=8.0,
                  save_path=None):
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])
    adi = integrate_nonisothermal(F0, T_in, P, W_cat, mode="adiabatic")
    cool = integrate_nonisothermal(F0, T_in, P, W_cat, mode="cooled",
                                   T_cool=T_cool, Uc=Uc)

    print("=" * 70)
    print(f"CHECK 4 -- axial profile & hotspot  (P={P:.0f} bar, W={W_cat:.0f} kg,"
          f" T_in={T_in:.0f} K)")
    print("=" * 70)
    for name, o in [("adiabatic", adi), (f"cooled (Uc={Uc:g})", cool)]:
        print(f"  {name:>18}:  T_max={o['T_max']:.1f} K "
              f"(+{o['dT_hot']:.1f} K at W={o['W_hot']:.1f} kg)   "
              f"T_out={o['T_out']:.1f} K   Y_MeOH={o['Y_MeOH']:.3f}")
    print(f"  Cu-sintering guardrail ~573 K (300 C): "
          f"{'OK' if cool['T_max'] < 573 else 'EXCEEDED'}\n")

    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        i = {s: k for k, s in enumerate(["CO2", "H2", "CH3OH", "H2O", "CO"])}
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)

        ax1.axhline(T_in, ls=":", color="grey", label="isothermal ref / coolant")
        ax1.plot(adi["W"], adi["T"], "-", color="#c0392b", label="adiabatic")
        ax1.plot(cool["W"], cool["T"], "-", color="#1f4e79",
                 label=f"cooled (Uc={Uc:g})")
        ax1.plot(cool["W_hot"], cool["T_max"], "o", color="#1f4e79")
        ax1.annotate(f"hotspot +{cool['dT_hot']:.1f} K",
                     (cool["W_hot"], cool["T_max"]),
                     textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax1.set_ylabel("Bed temperature (K)")
        ax1.set_title("Axial temperature profile "
                      "(single-pass CO$_2$ hydrogenation)")
        ax1.legend(frameon=False, fontsize=9)
        ax1.grid(alpha=0.3)

        ax2.plot(cool["W"], cool["F"][i["CH3OH"]], "-", color="#2e7d32",
                 label="CH$_3$OH")
        ax2.plot(cool["W"], cool["F"][i["CO"]], "-", color="#c55a11", label="CO")
        ax2.plot(cool["W"], cool["F"][i["H2O"]], "--", color="#2980b9",
                 label="H$_2$O")
        ax2.set_xlabel("Catalyst mass W (kg)")
        ax2.set_ylabel("Molar flow (mol/s)")
        ax2.set_title("Product build-up along the bed (cooled case)")
        ax2.legend(frameon=False, fontsize=9)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"Figure written to {save_path}\n")

    return adi, cool


# --------------------------------------------------------------------------- #
# Check 5 -- hotspot & yield vs load (feed-proportional turndown).
# --------------------------------------------------------------------------- #
def turndown_scan(P=50.0, W_cat=30.0, T_in=503.0, T_cool=503.0, Uc=8.0,
                  loads=None, save_path=None):
    if loads is None:
        loads = np.linspace(1.0, 0.2, 17)
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])
    Tmax, Whot, Ymeoh, Smeoh = [], [], [], []
    for phi in loads:
        o = integrate_nonisothermal(phi * F0, T_in, P, W_cat, mode="cooled",
                                    T_cool=T_cool, Uc=Uc)
        Tmax.append(o["T_max"]); Whot.append(o["W_hot"])
        Ymeoh.append(o["Y_MeOH"]); Smeoh.append(o["S_MeOH"])
    Tmax, Whot, Ymeoh, Smeoh = map(np.asarray, (Tmax, Whot, Ymeoh, Smeoh))

    print("=" * 70)
    print(f"CHECK 5 -- turndown  (cooled Uc={Uc:g}, T_cool={T_cool:.0f} K,"
          f" W={W_cat:.0f} kg)")
    print("=" * 70)
    print(f"{'load':>6}{'T_max[K]':>10}{'W_hot[kg]':>11}"
          f"{'S_MeOH':>9}{'Y_MeOH':>9}")
    for phi, tm, wh, s, y in zip(loads, Tmax, Whot, Smeoh, Ymeoh):
        print(f"{phi:>6.2f}{tm:>10.1f}{wh:>11.1f}{s:>9.3f}{y:>9.3f}")
    print()

    if save_path:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

        ax1.plot(100 * loads, Whot, "o-", color="#1f4e79")
        ax1.set_xlabel("Plant load (% of design feed)")
        ax1.set_ylabel("Hotspot location W_hot (kg)", color="#1f4e79")
        ax1b = ax1.twinx()
        ax1b.plot(100 * loads, Tmax, "s-", color="#c0392b")
        ax1b.set_ylabel("Peak temperature T_max (K)", color="#c0392b")
        ax1.set_title("Hotspot marches upstream as load drops")
        ax1.grid(alpha=0.3)

        ax2.plot(100 * loads, 100 * Ymeoh, "^-", color="#2e7d32",
                 label="MeOH yield")
        ax2.plot(100 * loads, 100 * Smeoh, "s-", color="#c55a11",
                 label="MeOH selectivity")
        ax2.set_xlabel("Plant load (% of design feed)")
        ax2.set_ylabel("Percent (%)")
        ax2.set_title("Per-pass performance improves at turndown")
        ax2.legend(frameon=False, fontsize=9)
        ax2.grid(alpha=0.3)

        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        print(f"Figure written to {save_path}\n")

    return loads, Tmax, Whot, Ymeoh, Smeoh


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    results = os.path.join(here, "results")
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
    check_nonisothermal_limit()
    axial_profile(save_path=os.path.join(results, "02_axial_hotspot.png"))
    turndown_scan(save_path=os.path.join(results, "03_turndown.png"))
