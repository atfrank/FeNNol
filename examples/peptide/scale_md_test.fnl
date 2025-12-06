# Scale MD test configuration for 1YCR peptide-protein system
# This tests the scaled MD method for studying unbinding kinetics

# System name
system = "1YCR_scale_md"

# Use ANI2x model (good for organic molecules)
model_file = "../md/ani2x.fnx"

# Device settings - use CPU for testing
device = cpu
double_precision = no

# Temperature and timestep
temperature = 300.0
dt = 0.5  # fs - use small timestep for stability

# Scale MD configuration
scale_md {
    # Input PDB file
    pdb_file = "1YCR_peptide_example.pdb"

    # Chain B is the peptide (chain of interest to track for unbinding)
    chain_of_interest = "B"

    # Alpha values to scan - 1.0 is unscaled, smaller values accelerate unbinding
    alpha_values = [0.1, 0.3, 0.5, 1.0]

    # Number of MD steps per alpha value
    nsteps_per_alpha = 5000

    # Thermostat settings
    thermostat = langevin
    gamma = 1.0  # friction in ps^-1

    # Fix backbone of protein (chain A) to reduce degrees of freedom
    fix_backbone {
        enabled = true
        chains = ["A"]
    }

    # Early stopping when peptide moves far enough
    early_stopping {
        enabled = true
        distance_threshold = 15.0  # Angstroms
        min_steps = 500           # Run at least this many steps
        check_frequency = 50      # Check distance every N steps
    }

    # Energy minimization before MD
    minimize = true
    min_steps = 200

    # Output settings
    output {
        trajectory_prefix = "1YCR_traj"
        distance_file = "1YCR_distances.dat"
    }
}

# Neighbor list settings
nblist_skin = 1.0
