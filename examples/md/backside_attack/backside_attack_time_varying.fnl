device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file sn2_reaction_initial.xyz  # Nucleophile and carbon initially far apart
  indexed no 
  has_comment_line yes
}

# Simulation parameters
nsteps = 500000  # Longer simulation for gradual approach
dt[fs] = 0.5
tdump[ps] = 5.0
nprint = 100
nsummary = 1000

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Restraints section with time-varying force constants
restraints {
  # Gradual backside attack restraint with increasing strength
  sn2_gradual_attack {
    type = backside_attack
    nucleophile = 15   # Index of attacking nucleophile (e.g., OH- oxygen)
    carbon = 5         # Index of carbon being attacked
    leaving_group = 12 # Index of leaving group (e.g., Cl-)
    target = 3.14159   # 180° linear approach
    
    # Time-varying force constant: gradually increase restraint strength
    force_constant = [0.1, 2.0, 10.0, 20.0]  # kcal/mol/rad²
    update_steps = 50000  # Change force constant every 50,000 steps
    style = harmonic
  }
  
  # Optional: Distance restraint to bring nucleophile closer initially
  approach_distance {
    type = distance
    atom1 = 15  # nucleophile
    atom2 = 5   # carbon
    target = 3.0  # Angstroms - initial approach distance
    
    # Start strong, then weaken as angle restraint takes over
    force_constant = [20.0, 10.0, 5.0, 1.0]  # kcal/mol/Å²
    update_steps = 50000
    style = flat_bottom
    tolerance = 0.5
  }
  
  # Example of one-sided restraint to prevent leaving group from getting too close
  prevent_recombination {
    type = distance
    atom1 = 5   # carbon
    atom2 = 12  # leaving group
    target = 2.5  # Minimum separation distance
    force_constant = 30.0
    style = one_sided
    side = lower  # Apply force only when distance < target
  }
}

# Collective variables to monitor reaction progress
colvars {
  # Primary reaction coordinate: backside attack angle
  attack_angle {
    type = angle
    atom1 = 15  # nucleophile
    atom2 = 5   # carbon
    atom3 = 12  # leaving group
  }
  
  # Secondary coordinates
  nucleophile_carbon_distance {
    type = distance
    atom1 = 15  # nucleophile
    atom2 = 5   # carbon
  }
  
  carbon_leaving_distance {
    type = distance
    atom1 = 5   # carbon
    atom2 = 12  # leaving group
  }
  
  # Reaction coordinate approximation
  # (distance difference: breaking bond - forming bond)
  # When negative, reaction is proceeding
  reaction_progress {
    type = distance  # This is simplified - real RC would be more complex
    atom1 = 5   # carbon
    atom2 = 15  # nucleophile (subtract this distance)
  }
}