# Simulated Annealing Example - Exponential Decay
#
# This example demonstrates simulated annealing with exponential temperature decay.
# Exponential decay is the most common schedule for simulated annealing.
#
# Use case: Finding low-energy conformations of molecules

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
nsteps = 50000
dt[fs] = 0.5
traj_format xyz

nblist_skin 2.

# Save trajectory
tdump[ps] = 1.0
nprint = 100
nsummary = 500

## Simulated Annealing Thermostat
thermostat ANNEAL

# Base temperature (required but will be overridden by annealing schedule)
temperature = 300.

# Friction constant for Langevin dynamics
gamma[THz] = 10.

## Annealing parameters
annealing{
  # Temperature range (in Kelvin)
  T_start = 800.0      # Start at high temperature (800 K)
  T_end = 50.0         # Cool down to near-frozen (50 K)

  # Annealing schedule type
  schedule = exponential   # Options: linear, exponential, cosine

  # Fraction of total steps to perform annealing
  # 1.0 = anneal over entire simulation
  # 0.8 = anneal over first 80%, then equilibrate at T_end
  anneal_steps = 1.0
}

# Expected behavior:
# - System starts at 800 K with high kinetic energy (explores configuration space)
# - Temperature decays exponentially: T(t) = T_end + (T_start - T_end) * exp(-t/tau)
# - System gradually cools and settles into low-energy conformations
# - Final 10,000 steps equilibrate at ~50 K
