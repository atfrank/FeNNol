# Quasi-Newton TS test with far initial guess
device = cpu
enable_x64 = True
matmul_prec = highest
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input {
    file = water_dimer_far.xyz
    indexed = no
    has_comment_line = yes
}

transition_state = True
ts_only = True
ts_method = quasi_newton
ts_hessian_update = bfgs
ts_trust_radius = 0.2
ts_eigenvalue_tolerance = 1e-4
ts_max_uphill_steps = 2
ts_initial_hessian_scale = 0.05

min_max_iterations = 30
min_force_tolerance = 1e-3
min_print_freq = 1
min_max_step = 0.3

output_prefix = quasi_newton_far
traj_format = xyz