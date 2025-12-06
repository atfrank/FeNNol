# Simulated Annealing for Conformer Search
#
# This example demonstrates using simulated annealing to find low-energy
# conformers of a small molecule (aspirin).
#
# Protocol:
# 1. Heat to high temperature (1000 K) to overcome rotation barriers
# 2. Slowly cool to 0 K to freeze in low-energy conformation
# 3. Use exponential decay for efficient exploration

device cuda:0
enable_x64
matmul_prec highest

model_file ../ani2x.fnx

xyz_input{
  file aspirin/aspirin.xyz
  indexed yes
  has_comment_line no
}

# Small molecule, no periodic boundaries needed
estimate_pressure no

# Simulation parameters
nsteps = 50000         # 25 ps total
dt[fs] = 0.5
traj_format xyz

nblist_skin 2.

# Save frequently to capture conformational changes
tdump[ps] = 0.5        # Every 0.5 ps
nprint = 50
nsummary = 500

## Simulated Annealing Thermostat
thermostat ANNEAL

temperature = 300.
gamma[THz] = 10.

## Aggressive annealing for conformer search
annealing{
  # Wide temperature range for thorough exploration
  T_start = 1000.0     # Very high T to overcome torsional barriers
  T_end = 10.0         # Near 0 K to freeze conformation

  # Exponential decay
  schedule = exponential

  # Anneal over entire simulation
  anneal_steps = 1.0
}

# Protocol notes:
# - High initial temperature (1000 K) allows bond rotations
# - Exponential cooling gradually reduces kinetic energy
# - System explores multiple conformers during cooling
# - Final low temperature (10 K) freezes molecule in stable conformation
# - Run multiple independent simulations with different random seeds to find global minimum
#
# Suggested workflow:
# 1. Run 10-20 independent annealing simulations
# 2. Extract final structures from each trajectory
# 3. Compare final energies to identify lowest-energy conformer
# 4. Further optimize found conformer with energy minimization
