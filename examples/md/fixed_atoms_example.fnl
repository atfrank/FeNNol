# Fixed Atoms Example Configuration
#
# This example demonstrates how to fix certain atoms during MD simulation.
# Fixed atoms don't move but still exert forces on mobile atoms (physically correct).
#
# Use cases:
#   - Active site calculations: Fix protein backbone, allow active site to move
#   - Solvation studies: Fix solute, allow solvent to equilibrate
#   - Large systems: Fix distant regions to speed up calculations

device cuda:0
model_file ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit Ha

# Input structure
xyz_input {
  file structure.xyz
  indexed no
  has_comment_line yes
}

# Simulation parameters
nsteps = 100000
dt[fs] = 0.5
tdump[ps] = 1.0
nprint = 100
nsummary = 1000

# Thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 1.0

###############################################################################
# FIXED ATOMS CONFIGURATION
###############################################################################

# Option 1: Select atoms to MOVE (everything else is fixed)
# --------------------------------------------------------
# fixed_atoms {
#   mode = mobile_residues   # Selection specifies what MOVES
#   pdb_file = protein.pdb
#
#   selection {
#     # Active site residues that can move
#     residue_numbers = 45, 46, "50-55", 100
#   }
# }

# Option 2: Select atoms to FIX (everything else moves)
# --------------------------------------------------------
# fixed_atoms {
#   mode = fixed_residues    # Selection specifies what's FIXED
#   pdb_file = protein.pdb
#
#   selection {
#     # Terminal regions are fixed
#     residue_numbers = "1-20", "180-200"
#   }
# }

# Option 3: Distance-based selection
# --------------------------------------------------------
# fixed_atoms {
#   mode = mobile_residues
#   pdb_file = protein.pdb
#
#   selection {
#     # Atoms within 10 Angstrom of active site can move
#     within_distance {
#       distance = 10.0
#       of {
#         residue_numbers = 50
#       }
#     }
#   }
# }

# Option 4: Direct atom indices (no PDB needed)
# --------------------------------------------------------
fixed_atoms {
  mode = fixed_residues
  # Fix first 10 atoms
  indices = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
}

# Option 5: Atom name based selection
# --------------------------------------------------------
# fixed_atoms {
#   mode = mobile_residues
#   pdb_file = protein.pdb
#
#   selection {
#     # Only backbone atoms can move
#     atom_names = CA, CB, C, N, O
#   }
# }

# Option 6: Chain-based selection
# --------------------------------------------------------
# fixed_atoms {
#   mode = mobile_residues
#   pdb_file = protein.pdb
#
#   selection {
#     # Only chain A can move
#     chain_ids = A
#   }
# }

###############################################################################
# SELECTION CRITERIA (can be combined in selection block):
#
#   residue_numbers  : List of residue numbers or ranges
#                      Examples: 1, 2, 3  or  "1-10", "50-60"
#
#   residue_names    : List of residue names
#                      Examples: ALA, GLY, PRO
#
#   atom_names       : List of atom names
#                      Examples: CA, CB, C, N, O, H
#
#   chain_ids        : List of chain identifiers
#                      Examples: A, B, C
#
#   within_distance  : Distance-based selection
#                      distance : cutoff in Angstrom
#                      of : nested selection criteria (what to measure from)
#
###############################################################################
# MODES:
#
#   fixed_residues   : Selection specifies atoms that are FIXED (default)
#   mobile_residues  : Selection specifies atoms that can MOVE
#
###############################################################################
