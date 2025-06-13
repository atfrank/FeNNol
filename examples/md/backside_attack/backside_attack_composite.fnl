device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file sn2_reaction.xyz  # SN2 reaction system
  indexed no 
  has_comment_line yes
}

# Simulation parameters
nsteps = 300000
dt[fs] = 0.5
tdump[ps] = 2.0
nprint = 100
nsummary = 1000

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Restraints section demonstrating composite backside attack restraint
restraints {
  # Composite backside attack restraint with both angle and distance components
  sn2_composite {
    type = backside_attack
    nucleophile = 15       # Index of attacking nucleophile
    carbon = 5             # Index of carbon being attacked
    leaving_group = 12     # Index of leaving group
    
    # Angle component (backside approach geometry)
    target = 3.14159                    # Target angle: 180° (π radians)
    angle_force_constant = 1.0 5.0 15.0  # Time-varying angle force constant
    
    # Distance component (bring nucleophile closer)
    target_distance = 2.5               # Target nucleophile-carbon distance (Å)
    distance_force_constant = 5.0 10.0 2.0  # Time-varying distance force constant
    
    update_steps = 50000                # Change force constants every 50,000 steps
    style = harmonic
  }
  
  # Alternative: Different time-varying profiles for angle vs distance
  sn2_independent_timing {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    leaving_group = 12
    
    # Angle component - gradual increase
    target = 3.14159
    angle_force_constant = 0.5 2.0 8.0 20.0
    
    # Distance component - early strong, then weaken
    target_distance = 3.0
    distance_force_constant = 15.0 10.0 5.0 1.0
    
    update_steps = 25000
    style = flat_bottom
    tolerance = 0.1  # ±5.7° for angle, ±0.1 Å for distance
  }
  
  # Simple composite restraint with constant force constants
  sn2_simple_composite {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    # leaving_group omitted - will auto-detect
    
    # Angle component
    target = 3.14159
    angle_force_constant = 10.0
    
    # Distance component
    target_distance = 2.8
    distance_force_constant = 8.0
    
    style = harmonic
  }
  
  # Angle-only restraint (traditional backside attack)
  sn2_angle_only {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    leaving_group = 12
    target = 3.14159
    angle_force_constant = 12.0
    # No distance component specified
    style = flat_bottom
    tolerance = 0.15
  }
}

# Collective variables to monitor both components
colvars {
  # Primary: backside attack angle
  attack_angle {
    type = angle
    atom1 = 15  # nucleophile
    atom2 = 5   # carbon
    atom3 = 12  # leaving group
  }
  
  # Primary: nucleophile-carbon distance
  nucleophile_distance {
    type = distance
    atom1 = 15  # nucleophile
    atom2 = 5   # carbon
  }
  
  # Secondary: carbon-leaving group distance
  leaving_distance {
    type = distance
    atom1 = 5   # carbon
    atom2 = 12  # leaving group
  }
  
  # Reaction coordinate approximation
  bond_difference {
    type = distance
    atom1 = 5   # This would ideally be calculated as
    atom2 = 15  # d(C-Leaving) - d(C-Nucleophile)
    # Note: Real reaction coordinate would require custom collective variable
  }
}