# Error handling test file (invalid input)

# Device and precision settings
device = cpu
enable_x64 = True

# Model file
model_file = /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

# System input - Invalid system to test error handling
xyz_input {
    inline = """3
    Invalid system for testing
    X   0.00000000   0.00000000   0.00000000
    Y   1.00000000   0.00000000   0.00000000
    Z   0.00000000   1.00000000   0.00000000
    """
}

# Transition state settings
transition_state = True
ts_only = True
ts_method = quasi_newton

# Minimization parameters
min_max_iterations = 5
min_force_tolerance = 1e-3
min_print_freq = 1