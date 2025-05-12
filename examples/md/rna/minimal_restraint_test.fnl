# Force CPU to avoid GPU memory issues
device cpu
enable_x64
matmul_prec highest
print_timings yes

model_file ./mace_mp_small.fnx

xyz_input{
  file minimal_test.xyz
  # whether the first column is the atom index (Tinker format)
  indexed no
  # whether a comment line is present
  has_comment_line yes
}

# compute neighborlists using minimum image convention
minimum_image no
# whether to wrap the atoms inside the first unit cell
wrap_box no
estimate_pressure no

# Very short test run
nsteps = 50
dt[fs] = 0.2
traj_format xyz

# Print frequently to see behavior
tdump[ps] = 0.001
nprint = 1
nsummary = 10

# Use NVE for cleaner testing
thermostat NVE

# Simple restraints on small system
restraints {
  # Distance restraint between carbon atoms
  c_c_restraint {
    type = distance
    atom1 = 0  # First C atom
    atom2 = 5  # Another C atom
    target = 3.0   # Target is 3 Å (current is 2 Å)
    force_constant = 100.0
    style = harmonic
  }
  
  # Flat-bottom restraint on O-O distance
  o_o_restraint {
    type = distance
    atom1 = 2  # First O atom
    atom2 = 7  # Another O atom
    target = 2.0
    force_constant = 100.0
    style = flat_bottom
    tolerance = 0.5  # Tolerate 2.0 ± 0.5 Å
  }
}

# Track these distances with colvars
colvars {
  c_c_distance {
    type = distance
    atom1 = 0
    atom2 = 5
  }
  
  o_o_distance {
    type = distance
    atom1 = 2
    atom2 = 7
  }
}