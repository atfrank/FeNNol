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

#0 cell vectors (format:  ax ay az bx by bz cx cy cz)
#cell = 62.23 0. 0. 0. 62.23 0. 0. 0. 62.23
# compute neighborlists using minimum image convention
minimum_image no
# whether to wrap the atoms inside the first unit cell
wrap_box no
estimate_pressure no

# number of steps to perform
nsteps = 100
# timestep of the dynamics
dt[fs] =  .5
traj_format xyz

nblist_skin 2.

#time between each saved frame
tdump[ps] = 0.01
# number of steps between each printing of the energy
nprint = 10
nsummary = 50
nblist_verbose

## set the thermostat
thermostat NVE 

## Thermostat parameters
temperature = 300.
#friction constant
gamma[THz] = 10.

# Restraints section
restraints {
  # Use atoms that definitely exist for debugging
  test_restraint {
    type = distance
    atom1 = 10  # First atom
    atom2 = 100  # Choose a different atom further away
    target = 10.0
    force_constant = 5.0
    style = harmonic
  }  
}

# Collective variables to track
colvars {
  debug_distance {
    type = distance
    atom1 = 10
    atom2 = 100
  } 
}