# Example of spherical boundary restraint for maintaining a spherical droplet
# This example simulates a water droplet with spherical boundary conditions

device cuda:0
enable_x64
matmul_prec highest

# Load model
model_file ../../../src/fennol/models/ani2x.fnx

# Input structure - water droplet
xyz_input{
  file watersmall/watersmall.xyz
  indexed no
  has_comment_line yes
}

# No periodic boundaries for droplet simulation
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

# Spherical boundary restraint
restraints {
  # Basic spherical boundary - keep all atoms inside sphere
  sphere_boundary {
    type = spherical_boundary
    center = 0.0, 0.0, 0.0  # Center at origin
    radius = 15.0           # 15 Angstrom radius
    force_constant = 10.0   # Harmonic force constant
    atoms = all             # Apply to all atoms
    mode = outside          # Keep atoms inside sphere
    style = harmonic        # Use harmonic restraint
  }
  
  # Alternative: Time-varying radius to compress/expand droplet
  # sphere_compress {
  #   type = spherical_boundary
  #   center = 0.0, 0.0, 0.0
  #   radius = 20.0, 18.0, 16.0, 14.0, 12.0  # Compress from 20 to 12 Angstroms
  #   force_constant = 5.0
  #   atoms = all
  #   mode = outside
  #   style = harmonic
  #   update_steps = 2000     # Change radius every 2000 steps
  #   interpolation_mode = interpolate  # Smooth radius change
  # }
  
  # Alternative: Apply to specific atoms only (e.g., surface atoms)
  # sphere_surface {
  #   type = spherical_boundary
  #   center = 0.0, 0.0, 0.0
  #   radius = 15.0
  #   force_constant = 20.0
  #   atoms = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9  # List specific atom indices
  #   mode = outside
  #   style = flat_bottom     # Only apply force beyond tolerance
  #   tolerance = 1.0         # 1 Angstrom tolerance before force kicks in
  # }
  
  # Alternative: Keep atoms outside a sphere (create a cavity)
  # sphere_cavity {
  #   type = spherical_boundary
  #   center = 0.0, 0.0, 0.0
  #   radius = 5.0
  #   force_constant = 50.0
  #   atoms = all
  #   mode = inside          # Keep atoms outside sphere (create cavity)
  #   style = one_sided      # One-sided restraint
  #   side = upper           # Apply force when atoms enter sphere
  # }
}

# Optional: Add collective variables to monitor sphere properties
colvars {
  # Monitor distance of specific atoms from center
  atom0_distance {
    type = sphere_distance
    atom_indices = 0
    center = 0.0, 0.0, 0.0
  }
  
  # You could also monitor center of mass, radius of gyration, etc.
}