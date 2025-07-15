# SN2 transition state test with MACE model

# Device and precision settings
device = cpu
enable_x64 = True
matmul_prec = highest

# Model file
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

# System input - Simple SN2 reaction: F- + CH3Cl -> FCH3 + Cl-
xyz_input {
    file = test_real_model/sn2_reaction.xyz
    indexed = no
    has_comment_line = yes
}

# Transition state settings
transition_state = True
ts_only = True
ts_method = sn2

# SN2 specific parameters
sn2_nu_index = 1      # F- (nucleophile)
sn2_c_index = 2       # C (carbon center)
sn2_lg_index = 6      # Cl- (leaving group)
sn2_target_nu_c_distance = 2.0
sn2_target_c_lg_distance = 2.0
sn2_constraint_strength = 0.05
sn2_reaction_coordinate_weight = 1.0
sn2_initial_step = 0.02

# Minimization parameters
min_max_iterations = 25
min_force_tolerance = 1e-3
min_print_freq = 1
min_max_step = 0.1

# Output settings
output_prefix = sn2_test
traj_format = xyz