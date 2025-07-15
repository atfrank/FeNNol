# Water dimer TS with quasi-Newton
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = water_dimer_ts.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = quasi_newton
ts_trust_radius = 0.15
ts_max_uphill_steps = 3
ts_initial_hessian_scale = 0.05

min_max_iterations = 25
min_force_tolerance = 1e-3
min_print_freq = 1
output_prefix = water_qn_test
traj_format = xyz