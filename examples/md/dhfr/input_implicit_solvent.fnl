device cuda:0
matmul_prec highest

model_file ../ani2x.fnx

xyz_input{
  file dhfr2_nowat.xyz
  # whether the first column is the atom index (Tinker format)
  indexed yes
  # whether a comment line is present
  has_comment_line no
}

# No periodic boundary conditions for implicit solvent
minimum_image no
wrap_box no
estimate_pressure no

# Implicit solvent (OBC Generalized Born)
implicit_solvent{
  model OBC
  dielectric 80.0
  cutoff 12.0
  surface_tension 0.005
  probe_radius 1.4
  radii_set mbondi
  include_nonpolar yes
}

# Short test run (1 ps = 2000 steps at 0.5 fs)
nsteps = 2000
# timestep of the dynamics
dt[fs] = 0.5
traj_format xyz

nblist_skin 2.

# time between each saved frame (save every 0.1 ps)
tdump[ps] = 0.1
# number of steps between each printing of the energy
nprint = 10
nsummary = 100
nblist_verbose

# Thermostat
thermostat LGV

# Thermostat parameters
temperature = 300.
# friction constant
gamma[THz] = 10.
