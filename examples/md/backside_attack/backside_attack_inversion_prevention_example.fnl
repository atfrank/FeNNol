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

# Restraints with inversion prevention
restraints {
  # Basic backside attack with automatic inversion prevention
  sn2_safe {
    type = backside_attack
    nucleophile = 5    # OH- oxygen
    carbon = 0         # Central carbon
    leaving_group = 4  # Chlorine
    
    # Angle constraint
    target = 180.0     # Degrees
    angle_force_constant = 15.0
    
    # Distance constraint (recommended to prevent wrong ordering)
    target_distance = 2.5         # Angstroms
    distance_force_constant = 10.0
    
    # Inversion prevention is ON by default
    # prevent_inversion = true    # Default: true
    # inversion_penalty_factor = 2.0  # Default: 2.0
    
    style = harmonic
  }
  
  # Example with custom inversion prevention settings
  sn2_custom_penalty {
    type = backside_attack
    nucleophile = 5
    carbon = 0
    leaving_group = 4
    
    target = 180.0
    angle_force_constant = 10.0
    
    # Stronger inversion penalty
    prevent_inversion = true
    inversion_penalty_factor = 3.0  # Stronger penalty (3x angle force constant)
    
    style = harmonic
  }
  
  # Example turning OFF inversion prevention (not recommended)
  sn2_no_prevention {
    type = backside_attack
    nucleophile = 5
    carbon = 0
    leaving_group = 4
    
    target = 180.0
    angle_force_constant = 20.0
    
    # Disable inversion prevention (may lead to Nu-LG-C arrangement)
    prevent_inversion = false
    
    style = harmonic
  }
  
  # Recommended: Composite restraint with strong distance control
  sn2_composite_safe {
    type = backside_attack
    nucleophile = 5
    carbon = 0
    leaving_group = 4
    
    # Angle and distance components
    target = 180.0
    angle_force_constant = 15.0
    target_distance = 2.5
    distance_force_constant = 20.0  # Strong distance control
    
    # Inversion prevention adds extra safety
    prevent_inversion = true
    inversion_penalty_factor = 2.5
    
    style = harmonic
  }
}

# Track the geometry to verify correct ordering
colvars {
  # Nu-C-LG angle (should approach 180°)
  attack_angle {
    type = angle
    atom1 = 5  # nucleophile
    atom2 = 0  # carbon (vertex)
    atom3 = 4  # leaving group
  }
  
  # Track all three distances
  nu_c_distance {
    type = distance
    atom1 = 5  # nucleophile
    atom2 = 0  # carbon
  }
  
  c_lg_distance {
    type = distance
    atom1 = 0  # carbon
    atom2 = 4  # leaving group
  }
  
  nu_lg_distance {
    type = distance
    atom1 = 5  # nucleophile
    atom2 = 4  # leaving group
  }
}