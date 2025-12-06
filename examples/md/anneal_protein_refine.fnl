# Simulated Annealing for Protein Structure Refinement
#
# This input file demonstrates a realistic protein structure refinement protocol
# using simulated annealing with implicit solvent.
#
# Protocol:
# 1. Start at elevated temperature (400 K) to escape local minima
# 2. Exponentially cool to physiological temperature (300 K)
# 3. Refine structure while maintaining proper folding

device cuda:0
enable_x64
matmul_prec highest

model_file ../ani2x.fnx

xyz_input{
  file dhfr/dhfr2_nowat.xyz
  indexed yes
  has_comment_line no
}

# Periodic boundary conditions
cell = 62.23 0. 0. 0. 62.23 0. 0. 0. 62.23
minimum_image yes
wrap_box no
estimate_pressure no

# Simulation parameters
nsteps = 200000        # 100 ps total
dt[fs] = 0.5
traj_format xyz

nblist_skin 2.

# Output settings
tdump[ps] = 10.0       # Save every 10 ps
nprint = 100
nsummary = 1000
nblist_verbose

## Simulated Annealing Thermostat
thermostat ANNEAL

# Base temperature (will be overridden)
temperature = 300.

# Langevin friction
gamma[THz] = 10.

## Annealing protocol for protein refinement
annealing{
  # Conservative temperature range to preserve structure
  T_start = 400.0      # High enough to explore, not so high as to denature
  T_end = 300.0        # Physiological temperature

  # Exponential decay (standard for protein refinement)
  schedule = exponential

  # Anneal over 75% of simulation, equilibrate for last 25%
  anneal_steps = 0.75
}

# Expected results:
# - Initial 150,000 steps: Temperature decays from 400 K to 300 K
# - System explores conformational space at high T
# - Gradual cooling allows structure to relax into stable conformation
# - Final 50,000 steps: Equilibrate at 300 K to assess stability
# - Energy should decrease and stabilize by end of simulation
