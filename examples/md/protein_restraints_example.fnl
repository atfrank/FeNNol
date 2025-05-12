device cuda:0
model_file ani2x.fnx  # Adjust model path as needed

traj_format arc
per_atom_energy no
energy_unit kcal/mol

xyz_input{
  file protein.xyz  # Your protein structure file
  indexed no 
  has_comment_line yes
}

# Simulation parameters
nsteps = 500000
dt[fs] = 0.5
tdump[ps] = 5.0
nprint = 100
nsummary = 1000

# Use Langevin thermostat at 300K
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

# Cell for periodic boundary conditions
cell = 50.0 0.0 0.0 0.0 50.0 0.0 0.0 0.0 50.0
minimum_image yes
wrap_box yes
nblist_skin 2.0

# Restraints section
restraints {
  # Backbone position restraints on CA atoms
  # These restrain selected alpha carbons to their initial positions
  ca_restraint1 {
    type = distance
    atom1 = 10  # CA atom index
    atom2 = 10  # Same atom - restrains to initial position
    target = 0.0
    force_constant = 5.0
    style = flat_bottom
    tolerance = 1.0  # Allow movement within 1 Å
  }
  
  ca_restraint2 {
    type = distance
    atom1 = 24  # Another CA atom
    atom2 = 24
    target = 0.0
    force_constant = 5.0
    style = flat_bottom
    tolerance = 1.0
  }
  
  # Distance restraint between specific residues
  # Example: restraining distance between two residues
  residue_distance {
    type = distance
    atom1 = 50  # First atom index
    atom2 = 150  # Second atom index
    target = 10.0  # Target distance in Angstroms
    force_constant = 2.0
    style = harmonic
  }
  
  # Angle restraint to maintain secondary structure
  helix_angle {
    type = angle
    atom1 = 30  # N atom
    atom2 = 32  # CA atom
    atom3 = 34  # C atom
    target = 1.91  # ~109.5 degrees in radians
    force_constant = 2.0
    style = flat_bottom
    tolerance = 0.15  # Allow ~8.5 degrees variation
  }
  
  # Dihedral restraint for specific torsion angle
  # Example: phi/psi angle in peptide backbone
  phi_dihedral {
    type = dihedral
    atom1 = 40  # C atom of previous residue
    atom2 = 42  # N atom 
    atom3 = 44  # CA atom
    atom4 = 46  # C atom
    target = -1.05  # ~-60 degrees in radians (alpha helix)
    force_constant = 1.0
    style = flat_bottom
    tolerance = 0.26  # Allow ~15 degrees variation
  }
}

# Collective variables to track
colvars {
  end_to_end_distance {
    type = distance
    atom1 = 5    # N-terminal atom
    atom2 = 200  # C-terminal atom
  }
  
  helix_angle1 {
    type = angle
    atom1 = 30
    atom2 = 32
    atom3 = 34
  }
  
  phi1 {
    type = dihedral
    atom1 = 40
    atom2 = 42
    atom3 = 44
    atom4 = 46
  }
  
  psi1 {
    type = dihedral
    atom1 = 42
    atom2 = 44
    atom3 = 46
    atom4 = 48
  }
}