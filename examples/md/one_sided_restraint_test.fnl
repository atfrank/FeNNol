# Test file for one-sided harmonic restraint
device cpu
enable_x64
matmul_prec highest

# System configuration
xyz examples/md/watersmall/watersmall.xyz
forcefield ani2x

# MD settings
integrator md
timestep 0.5
steps 100
output watersmall-one-sided-restraint
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

# Add restraints to test one-sided harmonic potential
restraints {
  # One-sided restraint - apply restraint only when O-O distance is less than 4.0 Å
  lower_side_test {
    type = distance
    atom1 = 0  # oxygen
    atom2 = 3  # oxygen 
    target = 4.0  # target distance in Å
    force_constant = 50.0
    style = one_sided
    side = lower  # apply force when distance < target (pulls up to target)
  }
  
  # One-sided restraint - apply restraint only when O-O distance is greater than 2.0 Å
  upper_side_test {
    type = distance
    atom1 = 6  # oxygen
    atom2 = 9  # oxygen
    target = 2.0  # target distance in Å
    force_constant = 50.0
    style = one_sided
    side = upper  # apply force when distance > target (pulls down to target)
  }
  
  # Regular harmonic restraint for comparison
  harmonic_test {
    type = distance
    atom1 = 12  # oxygen
    atom2 = 15  # oxygen
    target = 3.0  # target distance in Å
    force_constant = 50.0
    style = harmonic  # standard harmonic restraint
  }
}

# Add collective variables to track the distances
colvars {
  dist_lower_side {
    type = distance
    atom1 = 0
    atom2 = 3
  }
  
  dist_upper_side {
    type = distance
    atom1 = 6
    atom2 = 9
  }
  
  dist_harmonic {
    type = distance
    atom1 = 12
    atom2 = 15
  }
}