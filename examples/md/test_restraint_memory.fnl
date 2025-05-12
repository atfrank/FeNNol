# Minimal test file for restraint memory optimization
# Force CPU to avoid GPU memory issues
device cpu
enable_x64
matmul_prec highest

# No memory optimization features are used

# System configuration
xyz examples/md/watersmall/watersmall.xyz
forcefield ani2x

# MD settings
integrator md
timestep 0.5
steps 100
output watersmall-test-restraint
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

# Add a set of restraints
restraints {
  # Distance restraint
  hoh_1 {
    type = distance
    atom1 = 0  # first O
    atom2 = 3  # second O
    target = 3.0
    force_constant = 50.0
    style = harmonic
  }
  
  # Angle restraint
  hoh_angle_1 {
    type = angle
    atom1 = 1  # first H
    atom2 = 0  # first O
    atom3 = 2  # second H
    target = 1.91
    force_constant = 50.0
    style = harmonic
  }
  
  # Dihedral restraint (using analytical gradient)
  dihedral_1 {
    type = dihedral
    atom1 = 1  # H
    atom2 = 0  # O
    atom3 = 3  # O
    atom4 = 4  # H
    target = 0.0
    force_constant = 20.0
    style = harmonic
  }
}

# Add collective variables for monitoring
colvars {
  dist_o_o {
    type = distance
    atom1 = 0
    atom2 = 3
  }
  
  angle_hoh {
    type = angle
    atom1 = 1
    atom2 = 0
    atom3 = 2
  }
  
  dihedral_hooh {
    type = dihedral
    atom1 = 1
    atom2 = 0
    atom3 = 3
    atom4 = 4
  }
}