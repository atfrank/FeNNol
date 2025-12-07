# Test rigorous inter-chain force scaling
# This test compares alpha=0.0 (no inter-chain forces) vs alpha=1.0 (full forces)

system = "rigorous_test"

# Use ANI2x model
model_file = /home/user/FeNNol/examples/md/ani2x.fnx

# Device settings - CPU for testing
device = cpu
double_precision = no

# Temperature and timestep
temperature = 300.0
dt = 0.5

# Scale MD configuration
scale_md {
    pdb_file = "1YCR_peptide_example.pdb"
    chain_of_interest = "B"

    # Test with 2 alpha values
    alpha_values = 0.0 1.0
    nsteps_per_alpha = 100

    thermostat = langevin
    gamma = 1.0

    fix_backbone {
        enabled = true
        chains = A
    }

    early_stopping {
        enabled = false
    }

    # Use rigorous pairwise force decomposition
    rigorous_scaling = true

    # Enable force reporting to verify scaling
    report_inter_chain_forces = true
    report_frequency = 10

    # Quick minimization
    minimize = true
    min_steps = 20

    output {
        trajectory_prefix = rigorous_test_traj
        distance_file = rigorous_test_distances.dat
    }
}

nblist_skin = 1.0
