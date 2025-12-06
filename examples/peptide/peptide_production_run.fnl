# Peptide Scale MD Production Run
# Long simulation (~1 hour per alpha) with heating
# Tests force scaling effect on unbinding dynamics

system = "peptide_production"
model_file = /home/user/FeNNol/examples/md/ani2x.fnx
device = cpu
double_precision = no
temperature = 500.0
dt = 0.5

scale_md {
    pdb_file = "1YCR_peptide_example.pdb"
    chain_of_interest = "B"
    alpha_values = 0.0 1.0
    nsteps_per_alpha = 12000  # ~1 hour per alpha at ~5-6 steps/sec
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
        steps = 5000  # 2.5 ps heating
        fix_all_backbone = true
    }

    early_stopping {
        enabled = false
    }

    report_inter_chain_forces = true
    report_frequency = 500  # Write every 500 steps

    minimize = true
    min_steps = 200

    output {
        trajectory_prefix = production_traj
        distance_file = production_distances.dat
    }
}

nblist_skin = 1.0
