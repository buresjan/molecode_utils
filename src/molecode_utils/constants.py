# Unit conversions
HartreeToKcalMol = 627.5095  # 1 Hartree = 627.5095 kcal/mol
HartreeToeVolts = 27.2114079527  # 1 Hartree = 27.211407952subst_id eV
HartreeToJMol = 2625500  # 1 Hartree = 2625.5 kJ/mol = 2625500 J/mol

# Constants
G_solv = -260.2 / HartreeToKcalMol  # Hartree
R = 8.31446261815324  # J / mol
T = 298.15  # K
E_ref = 4.98  # eV
F = 96485.3321  # C / mol
k_B = 1.380649e-23  # J / K
h_Planck = 6.62607015e-34  # J s
molar_standard_state = 1.89432  # kcal / mol

# Derived unit conversions
VoltsToKcalMol = F / HartreeToJMol * HartreeToKcalMol
