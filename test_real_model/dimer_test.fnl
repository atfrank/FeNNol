# Dimer method transition state test with MACE model

# Device and precision settings
device = cpu
enable_x64 = True
matmul_prec = highest

# Model file
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

# System input - simple H2 dissociation
xyz_input {
    file = test_real_model/h2_dissociation.xyz
    indexed = no
    has_comment_line = yes
}

# Transition state settings
transition_state = True
ts_only = True
ts_method = dimer
dimer_separation = 0.01
dimer_rotation_tolerance = 0.1
dimer_max_rotations = 5
dimer_rotation_step = 0.1

# Minimization parameters
min_max_iterations = 15
min_force_tolerance = 1e-3
min_print_freq = 1
min_max_step = 0.1
min_initial_step = 0.01

# Output settings
output_prefix = dimer_test
traj_format = xyz