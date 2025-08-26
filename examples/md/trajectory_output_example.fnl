# Example demonstrating custom trajectory output paths
device cuda:0
model_file ../ani2x.fnx

xyz_input{
  file methyl_chloride.xyz
  indexed no 
  has_comment_line yes
}

# Simulation parameters
nsteps = 10000
dt[fs] = 0.5
traj_format xyz
tdump[ps] = 1.0
nprint = 100

thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# CUSTOM TRAJECTORY OUTPUT OPTIONS
# Option 1: Use output_prefix to specify prefix for all output files
output_prefix = results/sn2_simulation

# Option 2: Use trajectory_file to specify exact trajectory path (commented out)
# trajectory_file = /custom/path/my_trajectory

# The trajectory will be saved to:
# - With output_prefix: results/sn2_simulation.traj.xyz
# - With trajectory_file: /custom/path/my_trajectory.traj.xyz
# - Default (neither specified): methyl_chloride.traj.xyz

# Note: Make sure the output directory exists before running!
# FeNNol will not create missing directories.

restraints {
  # Example restraint
  c_cl_distance {
    type = distance
    atom1 = 0  # Carbon
    atom2 = 4  # Chlorine
    target = 1.8
    force_constant = 5.0
    style = harmonic
  }
}