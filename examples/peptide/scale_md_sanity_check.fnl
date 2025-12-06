# Scale MD Sanity Check Test
# Alpha = 0: Peptide should drift freely (no inter-chain forces)
# Alpha = 1: Normal MD (control case)

system = "sanity_check"

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

    # Test alpha = 0 (no inter-chain forces) and alpha = 1 (full forces)
    alpha_values = 0.0 1.0
    nsteps_per_alpha = 2000

    thermostat = langevin
    gamma = 1.0

    fix_backbone {
        enabled = true
        chains = A
    }

    early_stopping {
        enabled = false
    }

    # Enable force reporting to verify scaling
    report_inter_chain_forces = true
    report_frequency = 20

    # Minimization
    minimize = true
    min_steps = 100

    output {
        trajectory_prefix = sanity_check_traj
        distance_file = sanity_check_distances.dat
    }
}

nblist_skin = 1.0
