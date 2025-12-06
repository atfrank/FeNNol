# Water Dimer Long Test with Early Stopping
# Tests force scaling and early stopping functionality
# With alpha=0: Waters should drift apart and trigger early stopping
# With alpha=1: Waters should stay close (control)

system = "water_dimer_long"
model_file = /home/user/FeNNol/examples/md/ani2x.fnx
device = cpu
double_precision = no
temperature = 500.0  # Higher temp for faster diffusion
dt = 0.5

scale_md {
    pdb_file = "water_dimer.pdb"
    chain_of_interest = "B"
    alpha_values = 0.0 1.0
    nsteps_per_alpha = 20000  # 10 ps per alpha
    thermostat = langevin
    gamma = 1.0  # Lower friction for faster dynamics

    # No backbone fixing for water
    fix_backbone {
        enabled = false
    }

    # No heating for water test
    heating {
        enabled = false
    }

    # Early stopping - trigger when waters separate by 1.0 Angstroms
    early_stopping {
        enabled = true
        distance_threshold = 1.0  # Angstroms from initial COM
        min_steps = 500
        check_frequency = 100
    }

    report_inter_chain_forces = true
    report_frequency = 100

    minimize = true
    min_steps = 50

    output {
        trajectory_prefix = water_long_traj
        distance_file = water_long_distances.dat
    }
}

nblist_skin = 1.0
