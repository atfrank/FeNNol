# Multi-Site Backside Attack Restraint Example
# Simulating RNA selectivity for covalent inhibitor binding
# This example demonstrates competitive nucleophilic attack at multiple RNA sites

device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file rna_inhibitor.xyz  # RNA segment with reactive warhead
  indexed no 
  has_comment_line yes
}

# Simulation parameters
nsteps = 200000
dt[fs] = 0.5
tdump[ps] = 5.0
nprint = 500
nsummary = 2000

# Use Langevin thermostat
thermostat LGV
temperature = 310.0  # Physiological temperature
gamma[THz] = 1.0

# Multi-site backside attack restraint
restraints {
  rna_selectivity {
    type = multi_site_backside  # or backside_attack_multi
    
    # Multiple potential nucleophilic sites (e.g., 2'-OH groups, N7, N3)
    nucleophiles = [15, 23, 31, 45, 52]  # Atom indices of potential nucleophiles
    carbon = 5                           # Electrophilic carbon on warhead
    leaving_group = 12                   # Leaving group (optional - will auto-detect)
    
    # Target geometry for SN2 reaction
    target = 180.0                       # Target angle in degrees
    base_angle_force_constant = 3.0      # Base force for angle restraint
    
    # Optional distance component
    target_distance = 2.8                # Target Nu-C distance in Angstroms
    base_distance_force_constant = 2.0   # Base force for distance
    
    # Weighting configuration
    weighting_mode = hybrid              # "distance", "geometry", or "hybrid"
    distance_alpha = 0.4                 # Exponential decay factor for distance weighting
    standby_multiplier = 0.25            # Force multiplier for non-primary sites
    
    # Selectivity tracking
    write_selectivity = yes              # Track site preferences
    selectivity_output = rna_selectivity_data.dat
    
    style = harmonic
  }
  
  # Optional: Keep RNA structure stable
  rna_backbone {
    type = rmsd
    pdb_file = rna_reference.pdb
    atom_selection = "name P O3' O5' C3' C4' C5'"  # Backbone atoms
    target = 0.0
    force_constant = 50.0
    style = harmonic
  }
}

# Track reaction progress with collective variables
colvars {
  # Monitor reaction coordinates for each site
  rc_site1 {
    type = linear_combination
    coefficients = [1.0, -1.0]
    distances = [[15, 5], [5, 12]]  # RC = d(Nu1-C) - d(C-LG)
  }
  
  rc_site2 {
    type = linear_combination
    coefficients = [1.0, -1.0]
    distances = [[23, 5], [5, 12]]  # RC = d(Nu2-C) - d(C-LG)
  }
  
  rc_site3 {
    type = linear_combination
    coefficients = [1.0, -1.0]
    distances = [[31, 5], [5, 12]]  # RC = d(Nu3-C) - d(C-LG)
  }
  
  # Track individual distances
  dist_nu1_c {
    type = distance
    atom1 = 15
    atom2 = 5
  }
  
  dist_nu2_c {
    type = distance
    atom1 = 23
    atom2 = 5
  }
  
  dist_nu3_c {
    type = distance
    atom1 = 31
    atom2 = 5
  }
  
  # Track angles for geometry quality
  angle_nu1 {
    type = angle
    atom1 = 15  # Nu1
    atom2 = 5   # C
    atom3 = 12  # LG
  }
  
  angle_nu2 {
    type = angle
    atom1 = 23  # Nu2
    atom2 = 5   # C
    atom3 = 12  # LG
  }
}

# Analysis parameters
analysis {
  # Frequency for selectivity analysis
  selectivity_analysis_freq = 1000
  
  # Write detailed trajectory for visualization
  write_trajectory = yes
  trajectory_format = pdb
}