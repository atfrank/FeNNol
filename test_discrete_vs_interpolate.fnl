device cpu
model_file examples/md/ani2x.fnx

traj_format xyz
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  # Water molecule for simple test
  nxyz = 3
  coordinates[Angstrom] {
    O     0.0    0.0    0.0
    H     0.76   0.59   0.0
    H    -0.76   0.59   0.0
  }
}

# Very short simulation to clearly see the difference
nsteps = 3000
dt[fs] = 0.5
tdump[ps] = 0.1  # Frequent output to see transitions
nprint = 50      # Print every 50 steps
nsummary = 100

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Compare discrete vs interpolate modes side by side
restraints {
  # Discrete mode: Force constant jumps at steps 0, 1000, 2000
  discrete_OH_bond {
    type = distance
    atom1 = 0  # O
    atom2 = 1  # H
    
    target = 0.96  # Equilibrium O-H distance
    force_constant = 10.0 50.0 100.0  # Will jump: 10->50 at step 1000, 50->100 at step 2000
    
    update_steps = 1000
    interpolation_mode = discrete
    style = harmonic
  }

  # Interpolate mode: Force constant smoothly transitions
  interpolate_OH_bond {
    type = distance
    atom1 = 0  # O
    atom2 = 2  # other H
    
    target = 0.96  # Equilibrium O-H distance
    force_constant = 10.0 50.0 100.0  # Will smoothly transition between values
    
    update_steps = 1000
    interpolation_mode = interpolate
    style = harmonic
  }
}

# Monitor both O-H distances
colvars {
  OH1_distance {
    type = distance
    atom1 = 0  # O
    atom2 = 1  # H (discrete restraint)
  }
  
  OH2_distance {
    type = distance
    atom1 = 0  # O
    atom2 = 2  # H (interpolate restraint)
  }
  
  HOH_angle {
    type = angle
    atom1 = 1  # H
    atom2 = 0  # O
    atom3 = 2  # H
  }
}

# Enable debug output to see force constant values
restraint_debug yes