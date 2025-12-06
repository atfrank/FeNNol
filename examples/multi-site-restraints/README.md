# Multi-Site Restraints Example

This directory contains examples demonstrating the new multi-site backside attack restraint and preserve distances restraint features in FeNNol.

## Files

- `test_multi_site_rna.fnl` - Complete working example with both restraint types
- `focus_premod_mini.xyz` - Initial RNA structure with warhead
- `focus_premod_mini.pdb` - Reference PDB for RMSD restraints
- `rna_selectivity.dat` - Output file with selectivity tracking data (generated during simulation)

## Features Demonstrated

### 1. Multi-Site Backside Attack Restraint

Enables competitive nucleophilic attack studies where multiple sites can potentially react:

```fnl
multi_site_rna {
  type = multi_site_backside
  nucleophiles = 207, 175  # Multiple nucleophilic sites
  carbon = 229              # Electrophilic carbon
  leaving_group = 230       # Leaving group
  
  # Soft competition parameters
  weighting_mode = hybrid   # Distance, geometry, or hybrid weighting
  standby_multiplier = 0.3  # Force reduction for non-primary sites
  
  # Tracking
  write_selectivity = yes
  selectivity_output = rna_selectivity.dat
}
```

### 2. Preserve Distances Restraint

Prevents unwanted side reactions by maintaining specific atom-pair distances:

```fnl
prevent_side_reactions {
  type = preserve_distances
  
  # Flat list of atom pairs (atom1, atom2, atom3, atom4, ...)
  atom_pairs = 229, 228, 229, 230, 228, 227, ...
  
  force_constant = 1.0
  style = flat_bottom     # Only apply force when deviating
  tolerance = 0.2         # Allow 0.2Å flexibility
  
  # Distances auto-calculated from initial structure!
}
```

## Running the Example

```bash
fennol_md test_multi_site_rna.fnl
```

## Analyzing Results

The `rna_selectivity.dat` file contains timestep data showing:
- Which site is currently "primary" (highest weight)
- Reaction coordinates for each site
- Nu-C-LG angles for each site
- Distance metrics
- Weight distribution among sites

Example output:
```
# step  primary  RC_207  angle_207  dist_207  weight_207  RC_175  angle_175  dist_175  weight_175  total_E
1       207      5.639   133.7      7.133     0.733       5.947   97.7       7.441     0.267       1.242
```

## Key Advantages

1. **Natural Competition**: Sites compete based on their geometry and proximity
2. **No Hard Switching**: Smooth force transitions prevent discontinuities
3. **Prevents Side Reactions**: Preserve distances restraint maintains critical separations
4. **Comprehensive Tracking**: Full selectivity metrics for analysis

## Applications

- RNA selectivity for covalent inhibitors
- Protein cysteine competition studies
- DNA alkylation selectivity
- General competitive SN2 reaction modeling