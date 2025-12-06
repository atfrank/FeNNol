device cpu
double_precision
model_file examples/md/ani2x.fnx

xyz_input{
  file water_single.xyz
  indexed no
  has_comment_line yes
}

# No periodic boundary conditions for implicit solvent
minimum_image no
wrap_box no

# Implicit solvent parameters
implicit_solvent{
  model OBC
  dielectric 80.0
  cutoff 8.0
  radii_set mbondi
  include_nonpolar no
  charges = -0.834 0.417 0.417
}

# 1000 steps
nsteps = 1000
# timestep of the dynamics
dt[fs] = 0.5

# Trajectory output
traj_format xyz
# Save every 10 steps
tdump[fs] = 5.0

# Print energy every 10 steps
nprint = 10
nsummary = 1000

# Thermostat
thermostat LGV
temperature = 300.
# friction constant
gamma[THz] = 10.
