device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file sn2_reaction_system.xyz  # Your SN2 reaction system
  indexed no 
  has_comment_line yes
}

# Simulation parameters
nsteps = 100000
dt[fs] = 0.5
tdump[ps] = 2.0
nprint = 100
nsummary = 1000

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Restraints section with adaptive SN2
restraints {
  # Adaptive SN2 restraint - automatically configures all parameters
  sn2_auto {
    type = adaptive_sn2
    nucleophile = 5       # Index of attacking nucleophile
    carbon = 0            # Index of carbon being attacked
    # leaving_group = 4   # Optional - will auto-detect if not provided
    simulation_steps = 100000  # Total simulation steps (must match nsteps)
  }
}

# Collective variables to track the reaction progress
colvars {
  # Track the backside attack angle
  attack_angle {
    type = angle
    atom1 = 5  # nucleophile
    atom2 = 0  # carbon center
    atom3 = 4  # leaving group (adjust if auto-detected differently)
  }
  
  # Track C-nucleophile distance (forming bond)
  c_nucleophile_dist {
    type = distance
    atom1 = 0  # carbon
    atom2 = 5  # nucleophile
  }
  
  # Track C-leaving group distance (breaking bond)
  c_leaving_dist {
    type = distance
    atom1 = 0  # carbon
    atom2 = 4  # leaving group
  }
  
  # Track Nu-LG distance to monitor ordering
  nu_lg_dist {
    type = distance
    atom1 = 5  # nucleophile
    atom2 = 4  # leaving group
  }
}

# Enable restraint debugging to monitor adaptive behavior
restraint_debug = true