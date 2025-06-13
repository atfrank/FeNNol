
calculation:
  type: md
  steps: 10000
  
system:
  positions_file: molecule.arc
  
potential:
  type: gap
  
restraints:
  gradual_unfolding:
    type: rmsd
    reference_file: folded_structure.pdb
    target_rmsd: [0.2, 0.5, 1.0, 2.0]  # Gradually increase allowed RMSD
    force_constant: [10.0, 5.0, 2.0, 1.0]  # Decrease force constant over time
    mode: harmonic
    update_steps: 2500  # Change target every 2500 steps
    interpolation_mode: interpolate  # Smooth changes
    atoms: all
