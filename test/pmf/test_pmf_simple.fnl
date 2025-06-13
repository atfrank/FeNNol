device cpu
model_file examples/md/ani2x.fnx

traj_format xyz
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  # HCl molecule - simple system for PMF demonstration
  nxyz = 2
  coordinates[Angstrom] {
    H     0.0    0.0    0.0
    Cl    1.28   0.0    0.0  # Equilibrium H-Cl distance ~1.28 Å
  }
}

# Short simulation with multiple windows
nsteps = 15000
dt[fs] = 0.2      # Small timestep for light H atom
tdump[ps] = 0.5
nprint = 100
nsummary = 250

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 5.0  # Higher friction for better sampling

# PMF along H-Cl bond
restraints {
  hcl_bond_pmf {
    type = distance
    atom1 = 0  # H
    atom2 = 1  # Cl
    
    # Scan bond length from compressed to stretched
    target = 1.0 1.1 1.2 1.3 1.4 1.5 1.6  # Å
    force_constant = 100.0                  # Strong harmonic restraint
    
    update_steps = 2000                     # 2000 steps per window
    interpolation_mode = discrete            # Discrete windows for PMF
    estimate_pmf = yes                       # Enable PMF estimation
    equilibration_ratio = 0.25               # 25% equilibration
    
    style = harmonic
  }
}

# Monitor the bond
colvars {
  hcl_distance {
    type = distance
    atom1 = 0  # H
    atom2 = 1  # Cl
  }
}

# Enable debug to see sampling progress
restraint_debug yes

# PMF profile will show:
# - Minimum near equilibrium distance (~1.28 Å)
# - Rising potential for compression (< 1.28 Å)
# - Rising potential for extension (> 1.28 Å)
# Output: pmf_output.dat