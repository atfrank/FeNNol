
calculation:
  type: md
  steps: 1000
  
system:
  positions_file: molecule.arc
  
potential:
  type: gap
  
restraints:
  maintain_structure:
    type: rmsd
    target_rmsd: 0.5
    force_constant: 2.0
    mode: harmonic
    atoms: all  # or specify list like [0, 1, 2, 3]
