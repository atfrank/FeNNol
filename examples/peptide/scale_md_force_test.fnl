# Test Scale MD with inter-chain force reporting
system = test_forces

model_file = /home/user/FeNNol/examples/md/ani2x.fnx
device = cpu
double_precision = no

temperature = 300.0
dt = 0.5

scale_md {
    pdb_file = 1YCR_peptide_example.pdb
    chain_of_interest = B

    alpha_values = 0.5 1.0
    nsteps_per_alpha = 30

    thermostat = langevin
    gamma = 1.0

    fix_backbone {
        enabled = true
        chains = A
    }

    early_stopping {
        enabled = false
    }

    # Enable inter-chain force reporting
    report_inter_chain_forces = true

    minimize = true
    min_steps = 10

    output {
        trajectory_prefix = force_test_traj
        distance_file = force_test_distances.dat
    }
}

nblist_skin = 1.0
