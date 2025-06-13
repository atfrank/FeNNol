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

# Test time-varying target_distance
restraints {
  time_varying_distance_test {
    type = backside_attack
    nucleophile = 0       # Cl atom
    carbon = 1            # C atom
    leaving_group = 5     # Br atom
    
    # Fixed angle target
    target = 3.14159      # 180 degrees
    angle_force_constant = 10.0
    
    # Time-varying distance target: gradually decrease Nu-C distance
    target_distance = 2.5 2.4 2.3 2.1 2.0
    distance_force_constant = 5.0
    
    update_steps = 2000   # Change target distance every 2000 steps
    style = harmonic
  }
}

# Monitor the reaction progress
colvars {
  attack_angle {
    type = angle
    atom1 = 0  # Cl
    atom2 = 1  # C
    atom3 = 5  # Br
  }
  
  nu_c_distance {
    type = distance
    atom1 = 0  # Cl
    atom2 = 1  # C
  }
  
  c_lg_distance {
    type = distance  
    atom1 = 1  # C
    atom2 = 5  # Br
  }
}

# Enable debug output
restraint_debug yes