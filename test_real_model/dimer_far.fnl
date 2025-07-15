# Dimer method test with far initial guess
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = h2_dissociation_far.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = dimer
dimer_separation = 0.01
dimer_rotation_tolerance = 0.1
dimer_max_rotations = 10
dimer_rotation_step = 0.1

min_max_iterations = 30
min_force_tolerance = 1e-3
min_print_freq = 1
min_max_step = 0.3

output_prefix = dimer_far
traj_format = xyz