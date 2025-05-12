# Test file for time-varying force constants in restraints
device cpu
enable_x64
matmul_prec highest

# System configuration
xyz examples/md/watersmall/watersmall.xyz
forcefield ani2x

# MD settings
integrator md
timestep 0.5
steps 500  # Run for 500 steps to see the force constant changes
output watersmall-time-varying-restraint
log_stride 1
skin 2.0
rng_seed 12345
ensemble nvt

# Temperature control
thermostat langevin
temperature 300.0
tau 100.0

# PBC settings
cutoff 6.0
pbc

# Add restraints with time-varying force constants
restraints {
  # Distance restraint with gradually increasing force constant
  increasing_force {
    type = distance
    atom1 = 0  # oxygen 
    atom2 = 3  # oxygen
    target = 3.5  # target distance in Å
    force_constant = [0.0, 10.0, 20.0, 50.0, 100.0]  # Gradually increase from 0 to 100
    update_steps = 100  # Update force constant every 100 steps
    style = harmonic
  }
  
  # Distance restraint with oscillating force constant
  oscillating_force {
    type = distance
    atom1 = 6  # oxygen
    atom2 = 9  # oxygen
    target = 3.0  # target distance in Å
    force_constant = [50.0, 10.0, 50.0, 90.0, 50.0]  # Oscillate between high and low
    update_steps = 100  # Update force constant every 100 steps
    style = harmonic
  }
  
  # One-sided restraint with time-varying force constant
  one_sided_varying {
    type = distance
    atom1 = 12  # oxygen
    atom2 = 15  # oxygen
    target = 4.0  # target distance in Å
    force_constant = [0.0, 25.0, 50.0, 75.0, 100.0]  # Gradually increase
    update_steps = 100  # Update force constant every 100 steps
    style = one_sided
    side = lower  # Apply force when distance < target
  }
}

# Add collective variables to track the restrained distances
colvars {
  dist_increasing {
    type = distance
    atom1 = 0
    atom2 = 3
  }
  
  dist_oscillating {
    type = distance
    atom1 = 6
    atom2 = 9
  }
  
  dist_one_sided {
    type = distance
    atom1 = 12
    atom2 = 15
  }
}