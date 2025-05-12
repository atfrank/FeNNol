# Example input file for energy minimization

# General simulation parameters
device              = cpu            # Device to run on (cpu or gpu)
double_precision    = True          # Use double precision
matmul_prec         = highest        # Precision of matrix multiplication operations

# System 
coordinates         = watersmall/watersmall.xyz   # Input coordinates
model_type          = ani2x         # Model to use for energy/forces

# Minimization parameters
minimize            = True           # Enable minimization
minimize_only       = True           # Only perform minimization (no MD)
min_method          = lbfgs          # Minimization method: lbfgs, cg, sd
min_max_iterations  = 1000           # Maximum number of iterations
min_force_tolerance = 1e-4           # Force tolerance for convergence
min_energy_tolerance = 1e-6          # Energy tolerance for convergence
min_print_freq      = 1              # Print frequency
min_max_step        = 0.2            # Maximum step size (Angstroms)
min_history_size    = 10             # History size for L-BFGS

# Output settings
traj_format         = extxyz         # Trajectory format (xyz, extxyz, arc)