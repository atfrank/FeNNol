# H2 dissociation TS with dimer method
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = h2_ts_guess.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = dimer
dimer_separation = 0.01
dimer_rotation_tolerance = 0.1
dimer_max_rotations = 10

min_max_iterations = 20
min_force_tolerance = 1e-3
min_print_freq = 1
output_prefix = h2_dimer_test
traj_format = xyz