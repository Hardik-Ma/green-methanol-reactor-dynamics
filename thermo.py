"""
thermo.py -- Equilibrium constants and reaction enthalpies for the
CO2/CO/H2 -> methanol system on Cu/ZnO/Al2O3.

Equilibrium-constant correlations are the Graaf et al. (1986) expressions as
adopted by Vanden Bussche & Froment (1996), J. Catal. 161, 1-10.

    CONVENTION NOTE (read this before "fixing" anything):
    The 1996 paper reports  log10(K1*)   and  log10(1 / K3*).
        log10 K1*      =  3066/T - 10.592      (methanol reaction, bar^-2)
        log10 (1/K3*)  = -2073/T +  2.029      (RWGS)
    We define K_eq1 == K1*  and  K_eq2 == 1/K3*  so that BOTH rate brackets
    take the same "1 - Q/K_eq" approach-to-equilibrium form (see kinetics.py).
    Do not flip the sign of the K_eq2 exponent: K_eq2 is 1/K3*, by design.

Reaction convention:
    R1 (methanol): CO2 + 3 H2  <=> CH3OH + H2O    dH1 ~ -49 kJ/mol (exothermic)
    R2 (RWGS):     CO2 +   H2  <=> CO    + H2O    dH2 ~ +41 kJ/mol (endothermic)
"""

import numpy as np

# --- Reaction enthalpies (J/mol). Provisional, near-503 K lumped values. ---
# Only needed once the isothermal Stage-1 model is extended with the energy
# balance; unused in the isothermal validation. Refine against thermo data
# (e.g. via cp integration / Shomate) before the non-isothermal stage.
DH_MEOH = -49.0e3   # CO2 + 3 H2 -> CH3OH + H2O
DH_RWGS = +41.0e3   # CO2 +   H2 -> CO    + H2O


def K_eq1(T):
    """Methanol-reaction equilibrium constant K1* [bar^-2]. T in K."""
    return 10.0 ** (3066.0 / T - 10.592)


def K_eq2(T):
    """RWGS approach-to-equilibrium constant, defined as 1/K3* [dimensionless].

    T in K. See the CONVENTION NOTE above: this is 1/K3*, matching the form in
    which the RWGS bracket is written in kinetics.py.
    """
    return 10.0 ** (-2073.0 / T + 2.029)
