device cuda:0
double_precision
matmul_prec highest

model_file examples/md/ani2x.fnx

xyz_input{
  file examples/md/dhfr/dhfr2_nowat.xyz
  indexed yes
  has_comment_line no
}

# No periodic boundary conditions for implicit solvent
minimum_image no
wrap_box no
estimate_pressure no

# Implicit solvent (OBC Generalized Born)
# Note: Charges will be automatically estimated from atom types
# N: -0.3, O: -0.5, H: +0.3, C: +0.1, S: 0.0
implicit_solvent{
  model OBC
  dielectric 80.0
  cutoff 12.0
  radii_set mbondi
  include_nonpolar yes
}

# Benchmark run: 100 steps
nsteps = 100
dt[fs] = 0.5

# Output settings
traj_format xyz
tdump[fs] = 50.0  # Save every 100 steps (only 1 frame)
nprint = 10
nsummary = 100

nblist_skin 2.0

# Thermostat
thermostat LGV
temperature = 300.
gamma[THz] = 10.
