# Example input file for transition state optimization
# This example shows how to find a transition state using FeNNol

# General simulation parameters
device              = cpu            # Device to run on (cpu or gpu)
double_precision    = True          # Use double precision
matmul_prec         = highest        # Precision of matrix multiplication operations

# System 
coordinates         = ts_initial_guess.xyz   # Initial guess for transition state
model_type          = ani2x         # Model to use for energy/forces

# Transition state search parameters
transition_state    = True           # Enable transition state search
ts_only             = True           # Only perform TS search (no MD)
ts_method           = quasi_newton   # TS method: quasi_newton, dimer, or sn2

# Quasi-Newton specific parameters
ts_hessian_update   = bfgs          # Hessian update scheme: bfgs or sr1
ts_max_uphill_steps = 5             # Maximum consecutive uphill steps
ts_eigenvalue_tolerance = 1e-4      # Tolerance for negative eigenvalues
ts_trust_radius     = 0.3           # Trust radius in Angstroms
ts_initial_hessian_scale = -0.1     # Initial Hessian scaling

# Dimer method specific parameters (if using dimer method)
# dimer_separation    = 0.01          # Dimer separation in Angstroms
# dimer_rotation_tolerance = 0.1      # Rotation convergence tolerance
# dimer_max_rotations = 10            # Maximum rotation iterations
# dimer_rotation_step = 0.1           # Rotation step size
# dimer_initial_mode  = [0.1, 0.0, 0.0, ...]  # Initial dimer orientation (optional)

# General optimization parameters
min_max_iterations  = 500           # Maximum number of iterations
min_force_tolerance = 1e-4          # Force tolerance for convergence
min_print_freq      = 1             # Print frequency
min_max_step        = 0.2           # Maximum step size (Angstroms)

# Output settings
traj_format         = extxyz        # Trajectory format (xyz, extxyz, arc)
per_atom_energy     = True          # Display energy per atom
energy_unit         = kcal/mol      # Energy unit for display