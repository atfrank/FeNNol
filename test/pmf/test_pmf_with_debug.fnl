device cpu
model_file examples/md/ani2x.fnx

traj_format xyz
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  # Water molecule - simple system for PMF demo with detailed output
  nxyz = 3
  coordinates[Angstrom] {
    O     0.0    0.0    0.0
    H1    0.76   0.59   0.0
    H2   -0.76   0.59   0.0
  }
}

# Very short simulation to demonstrate debug output
nsteps = 6000
dt[fs] = 0.5
tdump[ps] = 0.5
nprint = 100
nsummary = 500  # PMF status will print every 500 steps

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 2.0

# PMF along O-H bond with detailed progress tracking
restraints {
  oh_bond_pmf {
    type = distance
    atom1 = 0  # O
    atom2 = 1  # H1
    
    # Scan O-H distance in 4 windows
    target = 0.85 0.96 1.07 1.18  # Å (compressed to stretched)
    force_constant = 50.0          # Same force constant for all windows
    
    update_steps = 1500            # 1500 steps per window
    interpolation_mode = discrete   # Required for PMF
    estimate_pmf = yes             # Enable PMF estimation
    equilibration_ratio = 0.2      # 20% equilibration (300 steps)
    
    style = harmonic
  }
}

# Monitor bonds
colvars {
  oh1_distance {
    type = distance
    atom1 = 0  # O
    atom2 = 1  # H1
  }
  
  oh2_distance {
    type = distance
    atom1 = 0  # O
    atom2 = 2  # H2
  }
  
  hoh_angle {
    type = angle
    atom1 = 1  # H1
    atom2 = 0  # O
    atom3 = 2  # H2
  }
}

# Enable debug output to see:
# - Window transitions
# - Equilibration progress
# - Sampling statistics
# - Real-time mean and standard deviation
# - Window completion summaries
restraint_debug yes

# At simulation end, you'll see:
# - Final PMF summary with all windows
# - Calculated PMF values
# - Barrier height and location
# - Output file: pmf_output.dat