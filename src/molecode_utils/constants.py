"""Common physical constants used by :mod:`molecode_utils`."""

# Unit conversions and physical constants used throughout the package.
# Conversion factors follow the standard literature values.
HartreeToKcalMol = 627.5095  # 1 Hartree = 627.5095 kcal/mol
HartreeToeVolts = 27.2114079527  # 1 Hartree = 27.2114079527 eV
HartreeToJMol = 2625500  # 1 Hartree = 2625.5 kJ/mol = 2625500 J/mol

# Constants
G_solv = -260.2 / HartreeToKcalMol  # Standard solvation free energy (Hartree)
R = 8.31446261815324        # Universal gas constant (J / mol K)
T = 298.15                  # Default temperature (K)
E_ref = 4.98                # Reference electrode potential (eV)
F = 96485.3321              # Faraday constant (C / mol)
k_B = 1.380649e-23          # Boltzmann constant (J / K)
h_Planck = 6.62607015e-34   # Planck constant (J s)
molar_standard_state = 1.89432  # Standard state correction (kcal / mol)

# Derived unit conversions
VoltsToKcalMol = F / HartreeToJMol * HartreeToKcalMol  # Conversion: 1 V → kcal/mol
