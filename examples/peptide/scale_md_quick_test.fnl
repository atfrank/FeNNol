# Minimal Scale MD test - quick CPU run
system = "1YCR_quick_test"

# Use ANI2x model
model_file = /home/user/FeNNol/examples/md/ani2x.fnx

# Device settings - CPU for testing
device = cpu
double_precision = no

# Temperature and timestep
temperature = 300.0
dt = 0.5

# Scale MD configuration - minimal for testing
scale_md {
    pdb_file = "1YCR_peptide_example.pdb"
    chain_of_interest = "B"

    # Just test with 2 alpha values, very few steps
    alpha_values = 0.5 1.0
    nsteps_per_alpha = 50

    thermostat = langevin
    gamma = 1.0

    fix_backbone {
        enabled = true
        chains = A
    }

    early_stopping {
        enabled = false
    }

    # Quick minimization
    minimize = true
    min_steps = 20

    output {
        trajectory_prefix = "quick_test_traj"
        distance_file = "quick_test_distances.dat"
    }
}

nblist_skin = 1.0
