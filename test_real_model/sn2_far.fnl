# SN2 method test with far initial guess
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = sn2_reaction_far.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = sn2
sn2_nu_index = 1
sn2_c_index = 2
sn2_lg_index = 6
sn2_target_nu_c_distance = 2.0
sn2_target_c_lg_distance = 2.0
sn2_constraint_strength = 0.05
sn2_reaction_coordinate_weight = 1.0
sn2_initial_step = 0.02

min_max_iterations = 30
min_force_tolerance = 1e-3
min_print_freq = 1
min_max_step = 0.1

output_prefix = sn2_far
traj_format = xyz