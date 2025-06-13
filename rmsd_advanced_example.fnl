
calculation:
  type: md
  steps: 5000
  
system:
  positions_file: molecule.arc
  
potential:
  type: gap
  
restraints:
  maintain_active_site:
    type: rmsd
    reference_file: reference_structure.pdb
    target_rmsd: 0.3
    force_constant: 5.0
    mode: flat_bottom  # Only apply force if RMSD > target
    atom_selection:
      residue_numbers: [25, 26, 27]  # Select specific residues
      atom_names: ["CA", "CB", "CG"]  # Select specific atom types
