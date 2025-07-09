# -------------------------------------------------------------------------
# Unified metadata for variables
# -------------------------------------------------------------------------
variable_metadata = {

    # ------------------------------------------------------------------ KEDs
    "KED_H": {
        "name": "Hydrogen Kinetic-Energy Distribution",
        "latex": "$\mathrm{KED}_\mathrm{H}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "KED_react_atoms": {
        "name": "Kinetic-Energy Distribution of Donor + Acceptor Atoms",
        "latex": "$\mathrm{KED}_{\mathrm{D+A}}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },

    # ------------------------------------------------ formation free energies
    "PC_formation_energy": {
        "name": "Product-Complex Formation Free Energy",
        "latex": "$w_\mathrm{P}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "RC_formation_energy": {
        "name": "Reactant-Complex Formation Free Energy",
        "latex": "$w_\mathrm{R}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ electronic / free energy
    "TS_energy": {
        "name": "Transition-State Electronic Energy",
        "latex": "$E_\mathrm{TS}$",
        "unit_latex": "$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "asynchronicity": {
        "name": "Asynchronicity",
        "latex": "$\eta$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "computed_barrier": {
        "name": "Computed Reaction Barrier",
        "latex": "$\Delta G^\ddagger$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "deltaG0": {
        "name": "Overall Reaction Free Energy",
        "latex": "$\Delta G^{0}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "deltaG0_inner": {
        "name": "Diagonal (Inner-Sphere) Reaction Free Energy",
        "latex": "$\Delta G^{0}_{\mathrm{diag}}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "frustration": {
        "name": "Thermodynamic Frustration",
        "latex": "$\sigma$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ tunnelling corrections
    "tunneling_corr_reaction": {
        "name": "Tunnelling Correction (Reaction)",
        "latex": "$\Delta G_\mathrm{tun}^\mathrm{rxn}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "tunneling_corr_self_reaction": {
        "name": "Tunnelling Correction (Self-Exchange)",
        "latex": "$\Delta G_\mathrm{tun}^\mathrm{self}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ electrochemical terms
    "E_H": {
        "name": "Proton-Coupled Reduction Potential of Abstractor",
        "latex": "$E^{\circ}_\mathrm{H}$",
        "unit_latex": "$\mathrm{V}$",
        "unit_name": "volt",
    },
    "E_ox_0": {
        "name": "Standard Reduction Potential of Oxidant",
        "latex": "$E^{\circ}_\mathrm{ox}$",
        "unit_latex": "$\mathrm{V}$",
        "unit_name": "volt",
    },
    "E_rad_deprot": {
        "name": "Radical/Deprotonated Reduction Potential",
        "latex": "$E_\mathrm{rad-deprot}$",
        "unit_latex": "$\mathrm{V}$",
        "unit_name": "volt",
    },

    # ------------------------------------------------ conceptual-DFT indices
    "mu": {
        "name": "Electronic Chemical Potential",
        "latex": "$\mu$",
        "unit_latex": "$\mathrm{eV}$",
        "unit_name": "electron-volt",
    },
    "omega": {
        "name": "Electrophilicity Index",
        "latex": "$\omega$",
        "unit_latex": "$\mathrm{eV}$",
        "unit_name": "electron-volt",
    },

    # ------------------------------------------------ acidity constants
    "pKaRH": {
        "name": "pKa of Substrate (RH)",
        "latex": "$\mathrm{p}K_\mathrm{a}(RH)$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "pKaRHplus": {
        "name": "pKa of Protonated Substrate (RH⁺)",
        "latex": "$\mathrm{p}K_\mathrm{a}(RH^+)$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },

    # ------------------------------------------------ self-exchange quantities
    "self_exchange_KED_H": {
        "name": "Self-Exchange Hydrogen KED",
        "latex": "\mathrm{KED}_\mathrm{H}^{\mathrm{self}}",
        "unit_latex": "-",
        "unit_name": "dimensionless",
    },
    "self_exchange_KED_react_atoms": {
        "name": "Self-Exchange Donor + Acceptor KED",
        "latex": "$\mathrm{KED}_{\mathrm{D+A}}^{\mathrm{self}}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "self_exchange_RC_formation": {
        "name": "Self-Exchange Reactant-Complex Formation Energy",
        "latex": "$w_\mathrm{R}^{\mathrm{self}}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "self_exchange_barrier": {
        "name": "Self-Exchange Reaction Barrier",
        "latex": "$\Delta G^\ddagger_\mathrm{self}$",
        "unit_latex": "$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ raw electronic energies
    "deprot": {
        "name": "Electronic Energy (Deprotonated)",
        "latex": "$E_\mathrm{deprot}$",
        "unit_latex": "$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "ox": {
        "name": "Electronic Energy (Oxidised)",
        "latex": "$E_\mathrm{ox}$",
        "unit_latex": "$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "rad": {
        "name": "Electronic Energy (Radical)",
        "latex": "$E_\mathrm{rad}$",
        "unit_latex": "$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "zero": {
        "name": "Electronic Energy (Neutral RH)",
        "latex": "$E_0$",
        "unit_latex": "$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },

    # ------------------------------------------------ metadata / identifiers
    "dataset": {
        "name": "Dataset Label",
        "latex": "$\mathrm{Dataset}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "molecule_id": {
        "name": "Molecule Identifier",
        "latex": "$\mathrm{ID}_\mathrm{mol}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "smiles": {
        "name": "SMILES String",
        "latex": "$\mathrm{SMILES}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },

    # ------------------------------------------------ atom-specific metadata
    "subst_target_atom": {
        "name": "Substrate Target Atom",
        "latex": "$\mathrm{Atom}_\mathrm{subst}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "subst_target_atom_other_hs": {
        "name": "Other H Atoms on Substrate Target",
        "latex": "$\mathrm{H}_\mathrm{other}^{\mathrm{subst}}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "oxid_target_atom": {
        "name": "Oxidant Target Atom",
        "latex": "$\mathrm{Atom}_\mathrm{oxid}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "oxid_target_atom_other_hs": {
        "name": "Other H Atoms on Oxidant Target",
        "latex": "$\mathrm{H}_\mathrm{other}^{\mathrm{oxid}}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "target_atom": {
        "name": "Generic Target Atom",
        "latex": "$\mathrm{Atom}_\mathrm{target}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
    "target_atom_other_hs": {
        "name": "Other H Atoms on Generic Target",
        "latex": "$\mathrm{H}_\mathrm{other}^{\mathrm{target}}$",
        "unit_latex": "$-$",
        "unit_name": "dimensionless",
    },
}
