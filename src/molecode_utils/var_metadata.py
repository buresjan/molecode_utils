# -------------------------------------------------------------------------
# Unified metadata for variables
# -------------------------------------------------------------------------
variable_metadata = {

    # ------------------------------------------------------------------ KEDs
    "KED_H": {
        "name": "Hydrogen Kinetic-Energy Distribution",
        "latex": r"$\mathrm{KED}_\mathrm{H}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "KED_react_atoms": {
        "name": "Kinetic-Energy Distribution of Donor + Acceptor Atoms",
        "latex": r"$\mathrm{KED}_{\mathrm{D+A}}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },

    # ------------------------------------------------ formation free energies
    "PC_formation_energy": {
        "name": "Product-Complex Formation Free Energy",
        "latex": r"$w_\mathrm{P}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "RC_formation_energy": {
        "name": "Reactant-Complex Formation Free Energy",
        "latex": r"$w_\mathrm{R}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ electronic / free energy
    "TS_energy": {
        "name": "Transition-State Electronic Energy",
        "latex": r"$E_\mathrm{TS}$",
        "unit_latex": r"$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "asynchronicity": {
        "name": "Asynchronicity",
        "latex": r"$\eta$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "computed_barrier": {
        "name": "Computed Reaction Barrier",
        "latex": r"$\Delta G^\ddagger$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "deltaG0": {
        "name": "Overall Reaction Free Energy",
        "latex": r"$\Delta G^{0}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "deltaG0_inner": {
        "name": "Diagonal (Inner-Sphere) Reaction Free Energy",
        "latex": r"$\Delta G^{0}_{\mathrm{diag}}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "frustration": {
        "name": "Thermodynamic Frustration",
        "latex": r"$\sigma$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ tunnelling corrections
    "tunneling_corr_reaction": {
        "name": "Tunnelling Correction (Reaction)",
        "latex": r"$\Delta G_\mathrm{tun}^\mathrm{rxn}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "tunneling_corr_self_reaction": {
        "name": "Tunnelling Correction (Self-Exchange)",
        "latex": r"$\Delta G_\mathrm{tun}^\mathrm{self}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ electrochemical terms
    "E_H": {
        "name": "Proton-Coupled Reduction Potential of Abstractor",
        "latex": r"$E^{\circ}_\mathrm{H}$",
        "unit_latex": r"$\mathrm{V}$",
        "unit_name": "volt",
    },
    "E_ox_0": {
        "name": "Standard Reduction Potential of Oxidant",
        "latex": r"$E^{\circ}_\mathrm{ox}$",
        "unit_latex": r"$\mathrm{V}$",
        "unit_name": "volt",
    },
    "E_rad_deprot": {
        "name": "Radical/Deprotonated Reduction Potential",
        "latex": r"$E_\mathrm{rad-deprot}$",
        "unit_latex": r"$\mathrm{V}$",
        "unit_name": "volt",
    },

    # ------------------------------------------------ conceptual-DFT indices
    "mu": {
        "name": "Electronic Chemical Potential",
        "latex": r"$\mu$",
        "unit_latex": r"$\mathrm{eV}$",
        "unit_name": "electron-volt",
    },
    "omega": {
        "name": "Electrophilicity Index",
        "latex": r"$\omega$",
        "unit_latex": r"$\mathrm{eV}$",
        "unit_name": "electron-volt",
    },

    # ------------------------------------------------ acidity constants
    "pKaRH": {
        "name": "pKa of Substrate (RH)",
        "latex": r"$\mathrm{p}K_\mathrm{a}(RH)$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "pKaRHplus": {
        "name": "pKa of Protonated Substrate (RH⁺)",
        "latex": r"$\mathrm{p}K_\mathrm{a}(RH^+)$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },

    # ------------------------------------------------ self-exchange quantities
    "self_exchange_KED_H": {
        "name": "Self-Exchange Hydrogen KED",
        "latex": r"\mathrm{KED}_\mathrm{H}^{\mathrm{self}}",
        "unit_latex": r"-",
        "unit_name": "dimensionless",
    },
    "self_exchange_KED_react_atoms": {
        "name": "Self-Exchange Donor + Acceptor KED",
        "latex": r"$\mathrm{KED}_{\mathrm{D+A}}^{\mathrm{self}}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "self_exchange_RC_formation": {
        "name": "Self-Exchange Reactant-Complex Formation Energy",
        "latex": r"$w_\mathrm{R}^{\mathrm{self}}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },
    "self_exchange_barrier": {
        "name": "Self-Exchange Reaction Barrier",
        "latex": r"$\Delta G^\ddagger_\mathrm{self}$",
        "unit_latex": r"$\mathrm{kcal/mol}$",
        "unit_name": "kcal/mol",
    },

    # ------------------------------------------------ raw electronic energies
    "deprot": {
        "name": "Electronic Energy (Deprotonated)",
        "latex": r"$E_\mathrm{deprot}$",
        "unit_latex": r"$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "ox": {
        "name": "Electronic Energy (Oxidised)",
        "latex": r"$E_\mathrm{ox}$",
        "unit_latex": r"$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "rad": {
        "name": "Electronic Energy (Radical)",
        "latex": r"$E_\mathrm{rad}$",
        "unit_latex": r"$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },
    "zero": {
        "name": "Electronic Energy (Neutral RH)",
        "latex": r"$E_0$",
        "unit_latex": r"$\mathrm{Hartree}$",
        "unit_name": "Hartree",
    },

    # ------------------------------------------------ metadata / identifiers
    "dataset": {
        "name": "Dataset Label",
        "latex": r"$\mathrm{Dataset}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "molecule_id": {
        "name": "Molecule Identifier",
        "latex": r"$\mathrm{ID}_\mathrm{mol}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "smiles": {
        "name": "SMILES String",
        "latex": r"$\mathrm{SMILES}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },

    # ------------------------------------------------ atom-specific metadata
    "subst_target_atom": {
        "name": "Substrate Target Atom",
        "latex": r"$\mathrm{Atom}_\mathrm{subst}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "subst_target_atom_other_hs": {
        "name": "Other H Atoms on Substrate Target",
        "latex": r"$\mathrm{H}_\mathrm{other}^{\mathrm{subst}}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "oxid_target_atom": {
        "name": "Oxidant Target Atom",
        "latex": r"$\mathrm{Atom}_\mathrm{oxid}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "oxid_target_atom_other_hs": {
        "name": "Other H Atoms on Oxidant Target",
        "latex": r"$\mathrm{H}_\mathrm{other}^{\mathrm{oxid}}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "target_atom": {
        "name": "Generic Target Atom",
        "latex": r"$\mathrm{Atom}_\mathrm{target}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
    "target_atom_other_hs": {
        "name": "Other H Atoms on Generic Target",
        "latex": r"$\mathrm{H}_\mathrm{other}^{\mathrm{target}}$",
        "unit_latex": r"$-$",
        "unit_name": "dimensionless",
    },
}
