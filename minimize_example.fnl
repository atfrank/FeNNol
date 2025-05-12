# Example input file for energy minimization

# General simulation parameters
device              = cpu            # Device to run on (cpu or gpu)
double_precision    = True           # Use double precision
matmul_prec         = highest        # Precision of matrix multiplication operations

# System parameters
coordinates         = examples/md/watersmall/watersmall.xyz   # Input coordinates
model_type          = ani2x          # Neural network model to use for energy/forces

# Minimization parameters
minimize            = True           # Enable minimization
minimize_only       = True           # Only perform minimization (no MD)
min_method          = lbfgs          # Minimization method: lbfgs, cg, sd
min_max_iterations  = 500            # Maximum number of iterations
min_force_tolerance = 1e-4           # Force tolerance for convergence (Hartree/Bohr)
min_energy_tolerance = 1e-6          # Energy tolerance for convergence (Hartree)
min_print_freq      = 10             # Print frequency for status updates
min_max_step        = 0.2            # Maximum step size (Angstroms)
min_history_size    = 20             # History size for L-BFGS

# Output settings
traj_format         = extxyz         # Trajectory format (xyz, extxyz, arc)