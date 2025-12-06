# Simulated Annealing Example - Linear Decay
#
# This example demonstrates simulated annealing with linear temperature decay.
# Linear decay provides uniform cooling rate.
#
# Use case: Systematic exploration with constant cooling rate

device cuda:0
enable_x64
matmul_prec highest

model_file ../ani2x.fnx

xyz_input{
  file watersmall/water27.xyz
  indexed yes
  has_comment_line no
}

# Simulation parameters
nsteps = 40000
dt[fs] = 0.5
traj_format xyz

nblist_skin 2.

# Save trajectory
tdump[ps] = 1.0
nprint = 100
nsummary = 500

## Simulated Annealing Thermostat
thermostat ANNEAL

temperature = 300.
gamma[THz] = 10.

## Annealing parameters
annealing{
  # Temperature range (in Kelvin)
  T_start = 600.0      # Start at 600 K
  T_end = 100.0        # Cool to 100 K

  # Linear decay schedule
  schedule = linear

  # Anneal over 80% of simulation, then equilibrate
  anneal_steps = 0.8
}

# Expected behavior:
# - Temperature decreases linearly from 600 K to 100 K over 32,000 steps
# - Constant cooling rate: dT/dt = (T_start - T_end) / anneal_steps
# - Last 8,000 steps (20%) maintain equilibrium at 100 K
