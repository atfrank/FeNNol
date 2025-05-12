# Force CPU to avoid GPU memory issues
device cpu
enable_x64
matmul_prec highest
print_timings yes

model_file ./mace_mp_small.fnx

xyz_input{
  file sub_system_premodified_epoxide_minimized_solv.xyz
  # whether the first column is the atom index (Tinker format)
  indexed no
  # whether a comment line is present
  has_comment_line yes
}

# compute neighborlists using minimum image convention
minimum_image no
# whether to wrap the atoms inside the first unit cell
wrap_box no
estimate_pressure no

# Shorter run with smaller step for stability
nsteps = 100  # Even fewer steps to avoid memory issues
dt[fs] = 0.2  # Slightly larger time step
traj_format xyz

# Print behavior but not too frequently to avoid memory issues
tdump[ps] = 0.002
nprint = 2
nsummary = 20

# Limit memory usage
nblist_stride = 10  # Force more frequent neighbor list updates

# Use NVE for cleaner testing
thermostat NVE

# Strong restraint to test effectivenesse
restraints {
  test_restraint {
    type = distance
    atom1 = 10
    atom2 = 100
    target = 5.0   # Round target distance
    force_constant = 500.0  # Very strong force constant
    style = harmonic
  }
  
  test_flat_restraint {
    type = distance
    atom1 = 20
    atom2 = 200
    target = 10.0
    force_constant = 500.0
    style = flat_bottom
    tolerance = 1.0
  }
}

# Track these distances with colvars
colvars {
  harmonic_distance {
    type = distance
    atom1 = 10
    atom2 = 100
  }
  
  flat_distance {
    type = distance
    atom1 = 20
    atom2 = 200
  }
}