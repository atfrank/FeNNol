# Example input file for SN2 transition state optimization
# This example shows how to find an SN2 transition state using FeNNol

# General simulation parameters
device              = cpu            # Device to run on (cpu or gpu)
double_precision    = True          # Use double precision
matmul_prec         = highest        # Precision of matrix multiplication operations

# System 
coordinates         = sn2_initial_guess.xyz   # Initial guess for SN2 TS
model_type          = ani2x         # Model to use for energy/forces

# SN2 Transition state search parameters
transition_state    = True           # Enable transition state search
ts_only             = True           # Only perform TS search (no MD)
ts_method           = sn2            # Use SN2-specific method

# SN2-specific parameters (1-based atom indices)
sn2_nu_index        = 1             # Index of nucleophile atom (Cl-)
sn2_c_index         = 2             # Index of carbon center
sn2_lg_index        = 8             # Index of leaving group (Br-)

# SN2 optimization parameters
sn2_target_nu_c_distance = 2.0      # Target Nu-C distance in Angstroms
sn2_target_c_lg_distance = 2.0      # Target C-LG distance in Angstroms
sn2_constraint_strength = 0.1       # Strength of geometric constraints
sn2_reaction_coordinate_weight = 1.0 # Weight for reaction coordinate following
sn2_initial_step    = 0.05          # Initial step size

# General optimization parameters
min_max_iterations  = 200           # Maximum number of iterations
min_force_tolerance = 1e-3          # Force tolerance for convergence (relaxed for SN2)
min_print_freq      = 1             # Print frequency
min_max_step        = 0.2           # Maximum step size (Angstroms)

# Output settings
traj_format         = extxyz        # Trajectory format (xyz, extxyz, arc)
per_atom_energy     = True          # Display energy per atom
energy_unit         = kcal/mol      # Energy unit for display