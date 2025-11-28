# Simulated Annealing Example - Cosine Decay
#
# This example demonstrates simulated annealing with cosine temperature decay.
# Cosine decay provides smooth cooling with slower initial rate.
#
# Use case: Gentle annealing for sensitive systems (proteins, biomolecules)

device cuda:0
enable_x64
matmul_prec highest

model_file ../ani2x.fnx

xyz_input{
  file dhfr/dhfr2_nowat.xyz  # Protein system
  indexed yes
  has_comment_line no
}

# Simulation parameters
nsteps = 100000
dt[fs] = 0.5
traj_format xyz

nblist_skin 2.

# Save trajectory every 5 ps
tdump[ps] = 5.0
nprint = 100
nsummary = 1000

## Simulated Annealing Thermostat
thermostat ANNEAL

temperature = 300.
gamma[THz] = 10.

## Annealing parameters
annealing{
  # Temperature range optimized for protein refinement
  T_start = 500.0      # Moderate high temperature to avoid denaturation
  T_end = 300.0        # Physiological temperature

  # Cosine decay: smooth and gentle
  schedule = cosine

  # Anneal over full simulation
  anneal_steps = 1.0
}

# Expected behavior:
# - Temperature follows cosine decay: T(t) = T_end + 0.5*(T_start-T_end)*(1+cos(π*t))
# - Slow cooling initially (allows thorough exploration)
# - Faster cooling in middle phase
# - Gradual approach to final temperature
# - Ideal for maintaining protein secondary structure during refinement
