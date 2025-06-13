device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file methyl_chloride.xyz  # Example: CH3Cl + OH- -> CH3OH + Cl-
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

# Restraints section demonstrating angle specification in degrees
restraints {
  # Backside attack restraint with angle in degrees
  sn2_backside_degrees {
    type = backside_attack
    nucleophile = 5    # Index of attacking oxygen atom (OH-)
    carbon = 0         # Index of carbon being attacked (CH3Cl)
    leaving_group = 4  # Index of chlorine leaving group
    target = 180.0     # Target angle in DEGREES (linear approach)
    force_constant = 10.0
    style = harmonic
  }
  
  # Example with different angle in degrees
  sn2_bent_approach {
    type = backside_attack
    nucleophile = 5    
    carbon = 0         
    leaving_group = 4  
    target = 170.0     # 170 degrees - slightly bent approach
    force_constant = 5.0
    style = flat_bottom
    tolerance = 0.1    # tolerance still in radians (~5.7 degrees)
  }
  
  # For comparison: traditional specification in radians
  sn2_radians {
    type = backside_attack
    nucleophile = 5    
    carbon = 0         
    target = 3.14159   # π radians = 180 degrees
    force_constant = 8.0
    style = harmonic
  }
  
  # Composite restraint with angle in degrees and distance
  sn2_composite_degrees {
    type = backside_attack
    nucleophile = 5
    carbon = 0
    target = 180.0              # Angle in degrees
    angle_force_constant = 10.0
    target_distance = 2.5       # Distance in Angstroms
    distance_force_constant = 5.0
    style = harmonic
  }
}

# Collective variables to track the reaction progress
colvars {
  # Track the backside attack angle
  attack_angle {
    type = angle
    atom1 = 5  # nucleophile (OH- oxygen)
    atom2 = 0  # carbon center
    atom3 = 4  # leaving group (Cl-)
  }
  
  # Track C-nucleophile distance (forming bond)
  c_nucleophile_dist {
    type = distance
    atom1 = 0  # carbon
    atom2 = 5  # nucleophile oxygen
  }
}