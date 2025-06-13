# Minimal example using time-varying target angle for SN2 reaction
# This configuration gradually changes the target angle from 140° to 180°

device cuda:0
model_file ../ani2x.fnx

xyz_input{
  file your_system.xyz  # Replace with your coordinate file
  indexed no 
  has_comment_line yes
}

# Basic MD settings
nsteps = 1000000
dt[fs] = 0.5
tdump[ps] = 5.0
nprint = 100

thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Restraints with time-varying angle
restraints {
  backside_attack_restraint {
    type = backside_attack
    nucleophile = 52
    carbon = 245
    leaving_group = 249
    
    # Gradually change from 140° to 180° over the simulation
    target = [140.0, 150.0, 160.0, 170.0, 180.0]
    
    # Gradually increase force constant too
    angle_force_constant = [0.008, 0.010, 0.012]
    
    # Change parameters every 200,000 steps
    update_steps = 200000
    
    # Use flat-bottom for stability
    style = flat_bottom
    tolerance = 0.2
    
    # Prevent wrong arrangement
    prevent_inversion = true
    inversion_penalty_factor = 1.0
    
    # Optional distance control
    target_distance = 5.0
    distance_force_constant = 0.01
  }
  
  # Supporting distance restraints
  keep_nu_c_close {
    type = distance
    atom1 = 52   # Nu
    atom2 = 245  # C
    target = 6.0
    force_constant = 0.02
    style = flat_bottom
    tolerance = 0.2
  }
  
  keep_nu_lg_far {
    type = distance
    atom1 = 52   # Nu
    atom2 = 249  # LG
    target = 7.0
    force_constant = 0.02
    style = flat_bottom
    tolerance = 0.2
  }
}