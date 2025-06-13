# Example of spherical boundary restraint with PDB-based atom selection
# This example demonstrates advanced atom selection using PDB information

device cuda:0
enable_x64
matmul_prec highest

# Load model
model_file ../../../src/fennol/models/ani2x.fnx

# Input structure - protein system
xyz_input{
  file dhfr/dhfr2.xyz
  indexed no
  has_comment_line yes
}

# No periodic boundaries for this example
minimum_image no
wrap_box no

# Simulation parameters
nsteps = 10000
dt[fs] = 0.5
traj_format xyz
tdump[ps] = 0.1
nprint = 100
nsummary = 1000

# Thermostat settings
thermostat Langevin
temperature = 300.0
gamma[THz] = 1.0

# Spherical boundary restraints with PDB-based selection
restraints {
  # Example 1: Keep protein compact using residue selection
  protein_sphere {
    type = spherical_boundary
    
    # Reference PDB file for atom selection
    pdb_file = dhfr/dhfr.pdb  # Assumes you have a PDB file
    
    # Auto-calculate center from selected atoms
    center = auto  # or specify: [x, y, z]
    
    # Atom selection criteria
    atom_selection {
      # Select backbone atoms of residues 10-50
      residue_numbers = 10, 11, 12, "15-30", "40-50"
      atom_names = CA, C, N, O  # Backbone atoms
    }
    
    radius = 20.0
    force_constant = 5.0
    mode = outside  # Keep atoms inside sphere
    style = flat_bottom
    tolerance = 2.0
  }
  
  # Example 2: Create cavity around active site
  active_site_cavity {
    type = spherical_boundary
    
    # Center on specific residue
    center = auto
    
    # Select atoms for center calculation
    atom_selection {
      residue_numbers = 121  # Active site residue
      atom_names = CA
    }
    
    # Apply restraint to water/solvent only
    atoms {
      residue_names = WAT, SOL  # Water molecules
    }
    
    radius = 8.0
    force_constant = 20.0
    mode = inside  # Keep atoms outside sphere (cavity)
    style = harmonic
  }
  
  # Example 3: Distance-based selection
  binding_pocket {
    type = spherical_boundary
    
    pdb_file = dhfr/dhfr.pdb
    center = auto
    
    # Select all atoms within 10 Å of residue 100
    atom_selection {
      within_distance {
        distance = 10.0
        of {
          residue_numbers = 100
        }
      }
    }
    
    radius = 15.0
    force_constant = 3.0
    mode = outside
    style = harmonic
  }
  
  # Example 4: Complex selection combining multiple criteria
  loop_region {
    type = spherical_boundary
    
    pdb_file = dhfr/dhfr.pdb
    center = auto
    
    # Select loop region atoms
    atom_selection {
      # Residues in loop
      residue_numbers = "45-55"
      # Only heavy atoms (exclude hydrogens)
      atom_names = C, CA, CB, CG, CD, CE, CZ, N, O, OG, OD1, OD2, OE1, OE2, NE, NH1, NH2, ND1, ND2, NZ, SD, SG
      # Only chain A
      chain_ids = A
    }
    
    # Time-varying radius to study loop dynamics
    radius = 12.0, 13.0, 14.0, 15.0, 14.0, 13.0, 12.0
    force_constant = 2.0
    update_steps = 1500
    interpolation_mode = interpolate
    mode = outside
    style = flat_bottom
    tolerance = 1.0
  }
  
  # Example 5: Using direct atom indices (backward compatibility)
  specific_atoms {
    type = spherical_boundary
    center = 10.0, 10.0, 10.0
    atoms = 100, 101, 102, 103, 104, 105  # Direct indices
    radius = 5.0
    force_constant = 10.0
    mode = outside
    style = harmonic
  }
}

# Monitor distances from sphere centers
colvars {
  # Monitor center of mass of selected region
  loop_com_distance {
    type = sphere_distance
    atom_indices = 450, 451, 452, 453, 454  # Some loop atoms
    center = auto  # Will use the auto-calculated center from loop_region restraint
  }
}