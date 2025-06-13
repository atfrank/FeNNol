device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file sn2_reaction_initial.xyz
  indexed no 
  has_comment_line yes
}

# Simulation parameters
nsteps = 1000000  # Long simulation for gradual angle change
dt[fs] = 0.5
tdump[ps] = 5.0
nprint = 100
nsummary = 1000

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Restraints section with time-varying target angle
restraints {
  # Backside attack restraint with gradually changing target angle
  # Starting at 140° and gradually increasing to 180°
  sn2_adaptive_angle {
    type = backside_attack
    nucleophile = 52     # Index of attacking nucleophile
    carbon = 245         # Index of carbon being attacked
    leaving_group = 249  # Index of leaving group
    
    # Time-varying target angle: gradually change from 140° to 180°
    target = [140.0, 150.0, 160.0, 170.0, 175.0, 180.0]  # degrees
    
    # Force constants for angle restraint (can also be time-varying)
    angle_force_constant = [0.008, 0.01, 0.012, 0.015]  # kcal/mol/rad²
    
    # Update angles and force constants every 200,000 steps
    update_steps = 200000
    
    style = flat_bottom
    tolerance = 0.2  # radians (~11.5 degrees)
    
    # Prevent inversion to avoid Nu-LG-C arrangement
    prevent_inversion = true
    inversion_penalty_factor = 1.0
    
    # Optional: Also control the Nu-C distance
    target_distance = 5.0  # Angstroms
    distance_force_constant = 0.01  # kcal/mol/Å²
  }
  
  # Additional distance restraints to maintain Nu-C-LG configuration
  # Ensure Nu-C distance is shorter than Nu-LG distance
  maintain_order1 {
    type = distance
    atom1 = 52   # nucleophile
    atom2 = 245  # carbon
    target = 6.0  # target distance in Å
    force_constant = 0.02
    style = flat_bottom
    tolerance = 0.2  # Allow movement within 0.2 Å
    update_steps = 500
  }
  
  maintain_order2 {
    type = distance
    atom1 = 52   # nucleophile
    atom2 = 249  # leaving group
    target = 7.0  # target distance in Å (larger than Nu-C)
    force_constant = 0.02
    style = flat_bottom
    tolerance = 0.2  # Allow movement within 0.2 Å
    update_steps = 500
  }
}

# Collective variables to monitor reaction progress
colvars {
  # Primary reaction coordinate: backside attack angle
  attack_angle {
    type = angle
    atom1 = 52   # nucleophile
    atom2 = 245  # carbon
    atom3 = 249  # leaving group
  }
  
  # Secondary coordinates
  nucleophile_carbon_distance {
    type = distance
    atom1 = 52   # nucleophile
    atom2 = 245  # carbon
  }
  
  carbon_leaving_distance {
    type = distance
    atom1 = 245  # carbon
    atom2 = 249  # leaving group
  }
  
  nucleophile_leaving_distance {
    type = distance
    atom1 = 52   # nucleophile
    atom2 = 249  # leaving group
  }
}