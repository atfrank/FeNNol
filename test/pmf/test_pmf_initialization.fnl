device cpu
model_file examples/md/ani2x.fnx

traj_format xyz
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  # Simple test system
  nxyz = 3
  coordinates[Angstrom] {
    O     0.0    0.0    0.0
    H1    0.96   0.0    0.0
    H2   -0.48   0.83   0.0
  }
}

# Short simulation to see initialization
nsteps = 3000
dt[fs] = 0.5
tdump[ps] = 1.0
nprint = 100
nsummary = 500

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 2.0

# PMF restraint with initialization messages
restraints {
  pmf_test {
    type = distance
    atom1 = 0  # O
    atom2 = 1  # H1
    
    target = 0.96
    force_constant = 20.0 40.0 60.0  # 3 windows
    
    update_steps = 1000
    interpolation_mode = discrete
    estimate_pmf = yes
    equilibration_ratio = 0.25  # 25% equilibration
    
    style = harmonic
  }
}

# Enable debug to see all messages
restraint_debug yes

# Expected output sequence:
# 1. During setup (immediately):
#    "PMF CONFIGURATION for restraint 'pmf_test':"
#    Shows all configuration details
#
# 2. When simulation starts (first force evaluation):
#    "PMF INITIALIZATION"
#    Shows PMF data structure setup
#
# 3. During simulation:
#    PMF progress messages every window