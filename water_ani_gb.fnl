device cpu
matmul_prec highest

model_file examples/md/ani2x.fnx

xyz_input{
  file water.xyz
  indexed no
  has_comment_line yes
}

# No periodic boundary conditions for implicit solvent
minimum_image no
wrap_box no
estimate_pressure no

# Implicit solvent (OBC Generalized Born)
implicit_solvent{
  model OBC
  dielectric 80.0
  cutoff 8.0
  surface_tension 0.005
  probe_radius 1.4
  radii_set mbondi
  include_nonpolar no
}

# MD settings
nsteps = 400
dt[fs] = 0.1
traj_format xyz

nblist_skin 0.5  # Small skin for single molecule
nblist_mult_size 2.0  # Increase neighbor list buffer
nblist_add_neigh 10  # Increase max neighbors for angle list

# Save every 0.01 ps
tdump[ps] = 0.01
nprint = 10
nsummary = 100
nblist_verbose

# Thermostat - Langevin
thermostat LGV
temperature = 300.
gamma[THz] = 10.
