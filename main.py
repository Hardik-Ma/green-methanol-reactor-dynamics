"""
main.py -- entry point for the green-methanol Stage 1 pipeline.

Put this file in the SAME folder as thermo.py, kinetics.py, reactor_ss.py,
and validate.py, then run:

    python main.py

It runs, in order:
    1. a base-case reactor evaluation,
    2. the equilibrium-vs-reactor validation table,
    3. the temperature sweep, writing results/01_temperature_sweep.png.
"""

import os

import numpy as np

from reactor_ss import integrate, _FLOOR
from validate import check_equilibrium, temperature_sweep


def main():
    # Write outputs into results/ next to this file, regardless of the
    # directory the script is launched from.
    here = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(here, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Feed order = kinetics.SPECIES: CO2, H2, CH3OH, H2O, CO.
    F0 = np.array([0.25, 0.75, _FLOOR, _FLOOR, _FLOOR])   # mol/s, H2/CO2 = 3

    print("Base case  T=503 K  P=50 bar  H2/CO2=3  W=30 kg")
    base = integrate(F0, T=503.0, P=50.0, W_cat=30.0)
    print(f"  X_CO2 = {base['X_CO2']:.3f}   S_MeOH = {base['S_MeOH']:.3f}"
          f"   Y_MeOH = {base['Y_MeOH']:.3f}\n")

    check_equilibrium([(483, 50), (503, 50), (523, 50), (503, 80)])

    temperature_sweep(
        P=50.0, W_cat=30.0,
        save_path=os.path.join(results_dir, "01_temperature_sweep.png"),
    )


if __name__ == "__main__":
    main()
