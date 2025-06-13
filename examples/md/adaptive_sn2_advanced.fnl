device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file complex_sn2_system.xyz  # Complex SN2 system with multiple atoms
  indexed no 
  has_comment_line yes
}

# Simulation parameters for longer reaction path
nsteps = 500000
dt[fs] = 0.5
tdump[ps] = 5.0
nprint = 500
nsummary = 5000

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 0.5  # Lower friction for smoother dynamics

# Restraints section with multiple adaptive SN2 restraints
restraints {
  # Primary adaptive SN2 restraint
  primary_sn2 {
    type = adaptive_sn2
    nucleophile = 52      # Index of primary nucleophile (e.g., OH-)
    carbon = 245          # Index of carbon being attacked
    leaving_group = 249   # Explicitly specify leaving group (e.g., Cl-)
    simulation_steps = 500000  # Match nsteps
  }
  
  # Optional: Additional restraints to prevent side reactions
  prevent_secondary {
    type = distance
    atom1 = 52    # nucleophile
    atom2 = 123   # some other reactive site
    target = 8.0  # Keep away from other sites
    force_constant = 0.1
    style = one_sided
    side = lower  # Only apply force if distance gets too small
  }
  
  # Stabilize spectator groups during reaction
  stabilize_spectator {
    type = angle
    atom1 = 250
    atom2 = 245  # carbon center
    atom3 = 251
    target = 1.911  # ~109.5 degrees in radians
    force_constant = 0.05
    style = flat_bottom
    tolerance = 0.2
  }
}

# Collective variables for detailed monitoring
colvars {
  # Primary reaction coordinate
  reaction_coord {
    type = angle
    atom1 = 52   # nucleophile
    atom2 = 245  # carbon
    atom3 = 249  # leaving group
  }
  
  # Bond distances
  nu_c_bond {
    type = distance
    atom1 = 52
    atom2 = 245
  }
  
  c_lg_bond {
    type = distance
    atom1 = 245
    atom2 = 249
  }
  
  # Reaction coordinate as distance difference
  reaction_progress {
    type = distance
    atom1 = 52
    atom2 = 249
  }
  
  # Monitor inversion prevention
  nu_lg_distance {
    type = distance
    atom1 = 52
    atom2 = 249
  }
}

# Enable restraint debugging
restraint_debug = true