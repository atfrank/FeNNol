# Peptide Scale MD Test with Heating
# Tests the heating phase before production runs

system = "peptide_heating_test"
model_file = /home/user/FeNNol/examples/md/ani2x.fnx
device = cpu
double_precision = no
temperature = 500.0  # Production temperature
dt = 0.5

scale_md {
    pdb_file = "1YCR_peptide_example.pdb"
    chain_of_interest = "B"
    alpha_values = 0.0 1.0
    nsteps_per_alpha = 500  # Short test
    thermostat = langevin
    gamma = 1.0

    fix_backbone {
        enabled = true
        chains = A  # Only fix chain A during production
    }

    # Heating phase - heat to 500 K with both backbones fixed
    heating {
        enabled = true
        target_temp = 500.0
        steps = 2000
        fix_all_backbone = true
    }

    early_stopping {
        enabled = false
    }

    report_inter_chain_forces = true
    report_frequency = 50

    minimize = true
    min_steps = 100

    output {
        trajectory_prefix = heating_test_traj
        distance_file = heating_test_distances.dat
    }
}

nblist_skin = 1.0
