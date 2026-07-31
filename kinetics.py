"""
kinetics.py -- LHHW kinetics for methanol synthesis + reverse water-gas shift
on a commercial Cu/ZnO/Al2O3 catalyst.

SOURCE (ground truth for the simulator):
    Vanden Bussche, K. M. & Froment, G. F. (1996).
    "A Steady-State Kinetic Model for Methanol Synthesis and the Water Gas
    Shift Reaction on a Commercial Cu/ZnO/Al2O3 Catalyst."
    Journal of Catalysis 161(1), 1-10.  doi:10.1006/jcat.1996.0156

Two independent reactions (rates in mol / kg_cat / s, partial pressures in bar):
    R1 (methanol): CO2 + 3 H2 <=> CH3OH + H2O
    R2 (RWGS):     CO2 +   H2 <=> CO    + H2O

Rate constants use the form   k_i = A_i * exp(B_i / (R * T)).
A_i, B_i are taken verbatim from Table 2 of the paper (VERIFIED against the
original, 2024): all five (A, B) pairs match. B has units J/mol; for the
lumped constant k1 (a product of a rate constant and several adsorption
equilibrium constants) B is a combination and is legitimately positive, whereas
for the RWGS rate constant k5, B = -E_a < 0, as the paper's Boudart checks
require.
"""

import numpy as np
import thermo

R = 8.314  # J / mol / K

# Species order used throughout the plant package.
SPECIES = ["CO2", "H2", "CH3OH", "H2O", "CO"]

# --- Table 2 parameters: (A, B) with k = A * exp(B / (R*T)) -------------------
#   k1 : lumped methanol rate constant   (r_MeOH prefactor)
#   k2 : coefficient of (pH2O / pH2)      in the adsorption denominator  [no T]
#   k3 : coefficient of sqrt(pH2)         in the adsorption denominator
#   k4 : coefficient of pH2O              in the adsorption denominator
#   k5 : RWGS rate constant               (r_RWGS prefactor)
_PARAMS = {
    "k1": (1.07,      36696.0),
    "k2": (3453.38,       0.0),   # no temperature dependence
    "k3": (0.499,     17197.0),
    "k4": (6.62e-11, 124119.0),
    "k5": (1.22e10,  -94765.0),
}


def _k(name, T):
    A, B = _PARAMS[name]
    return A * np.exp(B / (R * T))


def rates(p, T):
    """Reaction rates (r_MeOH, r_RWGS) in mol/kg_cat/s.

    Parameters
    ----------
    p : sequence of 5 partial pressures [bar], ordered as SPECIES
        (CO2, H2, CH3OH, H2O, CO).
    T : temperature [K].

    Returns
    -------
    (r_meoh, r_rwgs) : floats, mol/kg_cat/s.
    """
    pCO2, pH2, pMeOH, pH2O, pCO = p

    K1 = thermo.K_eq1(T)   # methanol,  = K1*
    K2 = thermo.K_eq2(T)   # RWGS,      = 1/K3*  (see thermo.CONVENTION NOTE)

    DEN = (1.0
           + _k("k2", T) * (pH2O / pH2)
           + _k("k3", T) * np.sqrt(pH2)
           + _k("k4", T) * pH2O)

    # Bracketed terms are approach-to-equilibrium factors (1 - Q/K):
    #   +1 far from equilibrium, 0 at equilibrium, <0 driving the reverse rxn.
    r_meoh = (_k("k1", T) * pCO2 * pH2
              * (1.0 - (pMeOH * pH2O) / (K1 * pH2**3 * pCO2))
              / DEN**3)

    r_rwgs = (_k("k5", T) * pCO2
              * (1.0 - (pCO * pH2O) / (K2 * pCO2 * pH2))
              / DEN)

    return r_meoh, r_rwgs
