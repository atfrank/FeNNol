device cuda:0
matmul_prec default  # Use default (float16 internally) instead of highest
print_timings no
model_file ../ani2x.fnx

xyz_input{
  file dhfr2.xyz
  indexed yes
  has_comment_line no
}

cell = 62.23 0. 0. 0. 62.23 0. 0. 0. 62.23
minimum_image yes
wrap_box no
estimate_pressure no

nsteps = 100
dt[fs] = .5
nblist_skin 1.5  # Reduced from 2.0 to decrease neighborlist size

traj_format xyz
nprint = 100
nsummary = 10000

thermostat LGV 

temperature = 300.
gamma[THz] = 10.

qtb{
  tseg[ps]=0.25
  omegacut[cm1]=15000.
  skipseg = 5
  startsave = 50
  agamma  = 1.
}
