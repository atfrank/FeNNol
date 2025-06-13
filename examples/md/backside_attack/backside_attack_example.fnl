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

# Restraints section
restraints {
  # Backside attack restraint
  # Forces the hydroxide oxygen (nucleophile) to approach the carbon 
  # from the opposite side of the chlorine (leaving group)
  sn2_backside_attack {
    type = backside_attack
    nucleophile = 5    # Index of attacking oxygen atom (OH-)
    carbon = 0         # Index of carbon being attacked (CH3Cl)
    leaving_group = 4  # Index of chlorine leaving group (optional - can be auto-detected)
    target = 180.0     # Target angle in degrees (for linear approach)
    force_constant = 10.0
    style = harmonic
  }
  
  # Optional: Additional distance restraint to bring nucleophile closer
  approach_distance {
    type = distance
    atom1 = 5  # hydroxide oxygen
    atom2 = 0  # carbon
    target = 2.5  # Angstroms - close enough for reaction
    force_constant = 5.0
    style = flat_bottom
    tolerance = 0.5
  }
  
  # Example with auto-detection of leaving group
  # (leaving_group parameter omitted - will find atom bonded to carbon furthest from nucleophile)
  sn2_auto_detect {
    type = backside_attack
    nucleophile = 5    # Index of attacking oxygen atom
    carbon = 0         # Index of carbon being attacked
    # leaving_group not specified - will be auto-detected
    target = 180.0     # 180 degrees
    force_constant = 5.0
    style = flat_bottom
    tolerance = 0.1    # Allow ±5.7 degrees deviation
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
  
  # Track C-leaving group distance (breaking bond)
  c_leaving_dist {
    type = distance
    atom1 = 0  # carbon
    atom2 = 4  # chlorine leaving group
  }
  
  # Track reaction coordinate (difference between breaking and forming bonds)
  reaction_coordinate {
    type = distance
    atom1 = 0  # This is a simplified example - real reaction coordinate
    atom2 = 5  # would require more sophisticated collective variables
  }
}