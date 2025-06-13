device cpu
model_file examples/md/ani2x.fnx

traj_format xyz
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  # Simple SN2 test system: Cl- + CH3Br -> CH3Cl + Br-
  nxyz = 8
  coordinates[Angstrom] {
    Cl   -3.5    0.0    0.0
    C     0.0    0.0    0.0  
    H     0.5    0.5    0.5
    H     0.5   -0.5   -0.5
    H    -0.5    0.5   -0.5
    Br    1.8    0.0    0.0
    H     0.0    0.0    2.0
    H     0.0    0.0   -2.0
  }
}

# Simulation parameters for PMF calculation
nsteps = 20000
dt[fs] = 0.5
tdump[ps] = 1.0
nprint = 100
nsummary = 500

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# PMF estimation along reaction coordinate
restraints {
  # PMF along Nu-C distance for SN2 reaction
  sn2_pmf {
    type = backside_attack
    nucleophile = 0       # Cl atom
    carbon = 1            # C atom
    leaving_group = 5     # Br atom
    
    # Keep angle fixed at 180 degrees
    target = 180.0        # degrees
    angle_force_constant = 20.0
    
    # Scan distance with discrete windows for PMF
    target_distance = 3.5 3.2 2.9 2.6 2.3 2.0  # Windows along reaction coordinate
    distance_force_constant = 15.0              # Strong restraint for good sampling
    
    update_steps = 3000                        # 3000 steps per window
    interpolation_mode = discrete               # Required for PMF estimation
    estimate_pmf = yes                          # Enable PMF calculation
    equilibration_ratio = 0.3                   # 30% equilibration, 70% sampling
    
    style = harmonic
  }
  
  # Alternative: PMF using simple distance restraint
  distance_pmf {
    type = distance
    atom1 = 0  # Cl
    atom2 = 1  # C
    
    target = 3.0                                # Fixed target
    force_constant = 10.0 15.0 20.0 15.0 10.0  # Vary force constant
    
    update_steps = 4000
    interpolation_mode = discrete
    estimate_pmf = yes
    equilibration_ratio = 0.25
    
    style = harmonic
  }
}

# Monitor the reaction
colvars {
  cl_c_distance {
    type = distance
    atom1 = 0  # Cl
    atom2 = 1  # C
  }
  
  c_br_distance {
    type = distance
    atom1 = 1  # C
    atom2 = 5  # Br
  }
  
  attack_angle {
    type = angle
    atom1 = 0  # Cl
    atom2 = 1  # C
    atom3 = 5  # Br
  }
  
  # Reaction coordinate (difference)
  reaction_coord {
    type = distance
    atom1 = 1  # C
    atom2 = 0  # Cl
    # Note: Ideal would be d(C-Br) - d(C-Cl) but using simple distance
  }
}

# Enable debug output to monitor PMF sampling
restraint_debug yes

# PMF output will be written to pmf_output.dat at the end of simulation