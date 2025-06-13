device cuda:0 
enable_x64
matmul_prec highest
print_timings yes

model_file /Users/afrank/Desktop/mace_mp_small.fnx

xyz_input{
  file /Users/afrank/Desktop/sub_system_premodified_epoxide_minimized_solv.xyz
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

# number of steps to perform
nsteps = 100
# timestep of the dynamics
dt[fs] =  0.5
traj_format xyz

# Shorter test run with more frequent output
tdump[ps] = 0.01
nprint = 10
nsummary = 50

## Using NVE for simplicity in testing
thermostat NVE

## Restraints section - pick two atoms that definitely exist and should be far apart
restraints {
  test_restraint {
    type = distance
    atom1 = 10  # Using 1-based indices in XYZ file, 0-based in code
    atom2 = 100  # These atoms should be different and far apart
    target = 10.0  # Target distance in Angstroms
    force_constant = 5.0  # Start with a moderate force constant
    style = harmonic
  }  
}

# Tracking the same atoms with colvars for verification
colvars {
  test_distance {
    type = distance
    atom1 = 10
    atom2 = 100
  } 
}