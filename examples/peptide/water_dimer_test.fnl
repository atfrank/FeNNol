# Water Dimer Force Scaling Test
# This tests if alpha scaling affects inter-molecular forces
# With alpha=0: Waters should drift apart (no inter-chain forces)
# With alpha=1: Waters should stay close (normal hydrogen bonding)

system = "water_dimer_test"
model_file = /home/user/FeNNol/examples/md/ani2x.fnx
device = cpu
double_precision = no
temperature = 300.0
dt = 0.5

scale_md {
    pdb_file = "water_dimer.pdb"
    chain_of_interest = "B"
    alpha_values = 0.0 1.0
    nsteps_per_alpha = 1000
    thermostat = langevin
    gamma = 10.0  # Strong friction for small system

    # No backbone fixing for water (no backbone atoms anyway)
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

    report_inter_chain_forces = true
    report_frequency = 10

    minimize = true
    min_steps = 100

    output {
        trajectory_prefix = water_test_traj
        distance_file = water_test_distances.dat
    }
}

nblist_skin = 1.0
