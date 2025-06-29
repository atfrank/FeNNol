device cpu
model_file examples/md/ani2x.fnx

traj_format xyz
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  # Simple 3-atom system
  nxyz = 3
  coordinates[Angstrom] {
    O     0.0    0.0    0.0
    H1    2.5    0.0    0.0  # Start at 2.5 Å
    H2   -0.5    0.0    0.0  # Leaving group
  }
}

# Short test simulation
nsteps = 1000
dt[fs] = 0.5
tdump[ps] = 1.0
nprint = 50
nsummary = 100

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 2.0

# Exact user configuration
restraints {
  backside_attack_restraint {
    type = backside_attack
    nucleophile = 1
    carbon = 0
    leaving_group = 2
    
    target = 180.0
    angle_force_constant = 0.0
    
    target_distance = 2.5, 2.4, 2.3, 2.1, 2.0
    distance_force_constant = 0.1
    
    interpolation_mode = discrete
    estimate_pmf = yes
    equilibration_ratio = 0.3
    update_steps = 300
    
    style = harmonic
  }
}

# Enable debug
restraint_debug yes