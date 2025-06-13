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

# Short simulation to demonstrate real-time output
nsteps = 3000
dt[fs] = 0.5
tdump[ps] = 1.0
nprint = 100
nsummary = 500

# Use Langevin thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 2.0

# PMF with real-time file output
restraints {
  pmf_with_realtime {
    type = distance
    atom1 = 0  # O
    atom2 = 1  # H1
    
    target = 0.96
    force_constant = 20.0 40.0 60.0  # 3 windows
    
    update_steps = 1000
    interpolation_mode = discrete
    estimate_pmf = yes
    write_realtime_pmf = yes      # Enable real-time file output
    equilibration_ratio = 0.25
    
    style = harmonic
  }
  
  # Regular PMF without real-time output for comparison
  pmf_without_realtime {
    type = distance
    atom1 = 0  # O
    atom2 = 2  # H2
    
    target = 0.96
    force_constant = 15.0 30.0 45.0
    
    update_steps = 1000
    interpolation_mode = discrete
    estimate_pmf = yes
    write_realtime_pmf = no       # No real-time output
    
    style = harmonic
  }
}

# Enable debug to see progress
restraint_debug yes

# Real-time files created:
# - pmf_pmf_with_realtime_realtime.dat (contains all samples as they're collected)
# - No real-time file for pmf_without_realtime
# 
# File format:
#    Step Win     Distance
#     250   0     0.963421
#     251   0     0.958734
#     ...