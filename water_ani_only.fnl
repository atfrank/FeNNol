device cpu
matmul_prec highest

model_file examples/md/ani2x.fnx

xyz_input{
  file water.xyz
  indexed no
  has_comment_line yes
}

minimum_image no
wrap_box no
estimate_pressure no

# MD settings
nsteps = 400
dt[fs] = 0.1
traj_format xyz

nblist_skin 0.5
nblist_mult_size 2.0
nblist_add_neigh 10

tdump[ps] = 0.01
nprint = 10
nsummary = 100

thermostat LGV
temperature = 300.
gamma[THz] = 10.
