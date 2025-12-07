# Water Dimer Asymmetric Force Scaling Test
# Tests the asymmetric scaling method:
# - Attractive inter-chain forces scaled by 1/alpha
# - Repulsive inter-chain forces scaled by alpha
# With alpha > 1: Should accelerate separation

system = "water_dimer_asymmetric_test"
model_file = /home/user/FeNNol/examples/md/ani2x.fnx
device = cpu
double_precision = no
temperature = 300.0
dt = 0.5

scale_md {
    pdb_file = "water_dimer.pdb"
    chain_of_interest = "B"
    alpha_values = 1.0 2.0 3.0
    nsteps_per_alpha = 500
    thermostat = langevin
    gamma = 10.0  # Strong friction for small system

    # No backbone fixing for water
    fix_backbone {
        enabled = false
    }

    # No heating for this quick test
    heating {
        enabled = false
    }

    early_stopping {
        enabled = false
    }

    # Use asymmetric scaling
    asymmetric_scaling = true

    report_inter_chain_forces = true
    report_frequency = 10

    minimize = true
    min_steps = 100

    output {
        trajectory_prefix = water_asymmetric_traj
        distance_file = water_asymmetric_distances.dat
    }
}

nblist_skin = 1.0
