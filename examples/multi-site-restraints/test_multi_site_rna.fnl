device cuda:0 
enable_x64
matmul_prec highest
print_timings yes
restraint_debug yes
nreplicas 1

model_file /home/aaron/ATX/projects/fennol-test/mace_mp_large.fnx

xyz_input{
  file /home/aaron/ATX/software/ATF-FeNNol/FeNNol/examples/multi-site-restraints/focus_premod_mini.xyz
  indexed no
  has_comment_line yes
}

minimum_image no
wrap_box no
estimate_pressure no

nsteps = 10000
dt[fs] = 0.5
traj_format xyz

nblist_skin 1.0

tdump[ps] = 0.005
nprint = 10
nsummary = 100
nblist_verbose

thermostat ADQTB 

temperature = 150.
gamma[THz] = 1.

qtb{
  tseg[ps]=0.25
  omegacut[cm1]=15000.
  skipseg = 5
  startsave = 50
  agamma = 1.
}


restraints {

   multi_site_rna {
      type = multi_site_backside              # or "backside_attack_multi"

      # Required: List of nucleophile atom indices (0-based)
      nucleophiles = 207, 175         # Your nucleophilic sites (e.g., 2'-OH, N7, N3)
      carbon = 229                              # Index of electrophilic carbon on warhead
      leaving_group = 230                      # Index of leaving group (optional - will auto-detect)

      # Target geometry
      target = 200.0                          # Target angle in degrees for backside attack
      base_angle_force_constant = 0.01        # Base force constant for angle restraint

      # Optional distance restraint
      target_distance = 1.7                   # Target Nu-C distance in Angstroms (optional)
      base_distance_force_constant = 0.1      # Force constant for distance (optional)

      # Weighting parameters
      weighting_mode = hybrid                 # "distance", "geometry", or "hybrid"
      distance_alpha = 0.5                    # Exponential decay factor (typically 0.3-0.7)
      standby_multiplier = 0.3                # Force multiplier for non-primary sites (0.2-0.4)

      # Selectivity tracking
      write_selectivity = yes                 # Enable selectivity data output
      selectivity_output = /home/aaron/ATX/software/ATF-FeNNol/FeNNol/examples/multi-site-restraints/rna_selectivity.dat # Output filename for tracking data

      # Restraint style
      style = harmonic                        # Restraint function type
    }

  topology_rmsd {
    type = rmsd
    target_rmsd = 0.0
    force_constant = 0.1
    reference_file = /home/aaron/ATX/software/ATF-FeNNol/FeNNol/examples/multi-site-restraints/focus_premod_mini.pdb
    atom_selection {
      residue_names = C, G, A, U
    }
    style = flat_bottom
    tolerance = 1.5
  }
  
    # NEW: Prevent side reactions by preserving distances
    prevent_side_reactions {
      type = preserve_distances

      # List of atom pairs to keep at current distances (flat list: atom1, atom2, atom3, atom4, ...)
      # This prevents unwanted nucleophiles from approaching
      atom_pairs = 229, 228, 229, 230, 228, 227, 227, 226, 226, 224, 227, 260, 227, 261, 228, 329, 229, 330, 229, 331

      force_constant = 1.0  # Adjust strength as needed
      style = flat_bottom     # Only apply force when getting too close
      tolerance = 0.2         # Allow 0.2Å flexibility

      # Target distances will be auto-calculated from initial structure
      # No need to specify them manually!
    }
 
      
}