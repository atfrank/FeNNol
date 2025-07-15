# Quasi-Newton transition state test with MACE model

# Device and precision settings
device = cpu
enable_x64 = True
matmul_prec = highest

# Model file
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

# System input
xyz_input {
    file = test_real_model/water_dimer.xyz
    indexed = no
    has_comment_line = yes
}

# Transition state settings
transition_state = True
ts_only = True
ts_method = quasi_newton
ts_hessian_update = bfgs
ts_trust_radius = 0.1
ts_eigenvalue_tolerance = 1e-4
ts_max_uphill_steps = 3

# Minimization parameters
min_max_iterations = 20
min_force_tolerance = 1e-3
min_print_freq = 1
min_max_step = 0.1

# Output settings
output_prefix = quasi_newton_test
traj_format = xyz