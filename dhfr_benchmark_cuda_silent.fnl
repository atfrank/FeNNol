device cuda:0
double_precision
matmul_prec highest

model_file examples/md/ani2x.fnx

xyz_input{
  file examples/md/dhfr/dhfr2_nowat.xyz
  indexed yes
  has_comment_line no
}

# No periodic boundary conditions
minimum_image no
wrap_box no
estimate_pressure no

# Implicit solvent (OBC Generalized Born)
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

# MINIMAL OUTPUT - reduce I/O overhead
traj_format xyz
tdump[fs] = 10000.0  # Never save (>100 steps)
nprint = 50          # Print twice only
nsummary = 100       # Only final summary

nblist_skin 2.0

# Thermostat
thermostat LGV
temperature = 300.
gamma[THz] = 10.
