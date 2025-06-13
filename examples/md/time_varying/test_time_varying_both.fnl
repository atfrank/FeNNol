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

# Test both time-varying target angle and distance
restraints {
  time_varying_both_test {
    type = backside_attack
    nucleophile = 0       # Cl atom
    carbon = 1            # C atom
    leaving_group = 5     # Br atom
    
    # Time-varying angle target: gradually approach 180 degrees
    target = 150.0 160.0 170.0 175.0 180.0  # degrees
    angle_force_constant = 5.0 10.0 15.0 20.0 25.0
    
    # Time-varying distance target: gradually decrease Nu-C distance
    target_distance = 3.0 2.8 2.6 2.4 2.2
    distance_force_constant = 2.0 4.0 6.0 8.0 10.0
    
    update_steps = 2000   # Change targets every 2000 steps
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