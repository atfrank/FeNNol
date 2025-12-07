# Water Dimer Asymmetric Scaling - Longer Test
# Run longer to see actual separation

system = "water_dimer_asymmetric_long"
model_file = /home/user/FeNNol/examples/md/ani2x.fnx
device = cpu
double_precision = no
temperature = 300.0
dt = 0.5

scale_md {
    pdb_file = "water_dimer.pdb"
    chain_of_interest = "B"
    alpha_values = 1.0 2.0 3.0
    nsteps_per_alpha = 5000  # 2.5 ps per alpha
    thermostat = langevin
    gamma = 2.0  # Lower friction for faster dynamics

    fix_backbone {
        enabled = false
    }

    heating {
        enabled = false
    }

    early_stopping {
        enabled = true
        distance_threshold = 2.0  # Stop when separated by 2 Angstroms
        min_steps = 500
        check_frequency = 100
    }

    # Use asymmetric scaling
    asymmetric_scaling = true

    report_inter_chain_forces = true
    report_frequency = 50

    minimize = true
    min_steps = 100

    output {
        trajectory_prefix = water_asym_long_traj
        distance_file = water_asym_long_distances.dat
    }
}

nblist_skin = 1.0
