device cpu
model_file examples/md/ani2x.fnx

traj_format xyz
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  # Simple SN2 test system: Cl- + CH3Br -> CH3Cl + Br-
  nxyz = 8
  coordinates[Angstrom] {
    Cl   -3.0    0.0    0.0
    C     0.0    0.0    0.0  
    H     0.5    0.5    0.5
    H     0.5   -0.5   -0.5
    H    -0.5    0.5   -0.5
    Br    1.8    0.0    0.0
    H     0.0    0.0    2.0
    H     0.0    0.0   -2.0
  }
}

# Short simulation for testing
nsteps = 10000
dt[fs] = 0.5
tdump[ps] = 0.5
nprint = 100
nsummary = 500

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Test different interpolation modes
restraints {
  # Test 1: Interpolated mode (smooth transitions) - DEFAULT
  interpolated_distance {
    type = backside_attack
    nucleophile = 0       # Cl atom
    carbon = 1            # C atom
    leaving_group = 5     # Br atom
    
    # Fixed angle
    target = 3.14159      # 180 degrees
    angle_force_constant = 10.0
    
    # Time-varying distance with interpolation
    target_distance = 3.0 2.5 2.0
    distance_force_constant = 5.0 10.0 15.0
    
    update_steps = 3000
    interpolation_mode = interpolate  # Smooth linear interpolation
    style = harmonic
  }

  # Test 2: Discrete mode (step changes)
  discrete_angle {
    type = distance
    atom1 = 0  # Cl
    atom2 = 1  # C
    
    target = 2.5  # Fixed target
    force_constant = 1.0 5.0 10.0 20.0  # Jump between values
    
    update_steps = 2500
    interpolation_mode = discrete  # Step changes at boundaries
    style = harmonic
  }

  # Test 3: Backside attack with discrete angle changes
  discrete_backside {
    type = backside_attack
    nucleophile = 0
    carbon = 1
    leaving_group = 5
    
    # Discrete angle changes
    target = 150.0 165.0 180.0  # degrees - will jump between values
    angle_force_constant = 5.0
    
    # Interpolated distance (mixing modes)
    target_distance = 3.0 2.0
    distance_force_constant = 8.0
    
    update_steps = 3300
    interpolation_mode = discrete  # Applies to all time-varying parameters
    style = harmonic
  }

  # Test 4: Smooth transitions for reaction path
  smooth_reaction_path {
    type = backside_attack
    nucleophile = 0
    carbon = 1
    leaving_group = 5
    
    # Smoothly varying angle
    target = 140.0 160.0 170.0 180.0  # degrees
    angle_force_constant = 2.0 5.0 10.0 15.0
    
    # Smoothly varying distance
    target_distance = 3.5 3.0 2.5 2.2
    distance_force_constant = 3.0 6.0 9.0 12.0
    
    update_steps = 2500
    interpolation_mode = interpolate  # Smooth transitions
    style = flat_bottom
    tolerance = 0.1
  }
}

# Monitor the changes
colvars {
  cl_c_distance {
    type = distance
    atom1 = 0  # Cl
    atom2 = 1  # C
  }
  
  attack_angle {
    type = angle
    atom1 = 0  # Cl
    atom2 = 1  # C
    atom3 = 5  # Br
  }
}

# Enable debug output to see the transitions
restraint_debug yes