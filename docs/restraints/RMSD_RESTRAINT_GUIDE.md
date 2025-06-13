# RMSD Restraint Implementation Guide

## Overview
The RMSD (Root Mean Square Deviation) restraint maintains structural similarity to a reference configuration by applying forces when the current structure deviates from a target RMSD value.

## Features

### 1. Performance Optimizations
- **JAX JIT compilation** for core calculations
- **Vectorized operations** for efficient force computation
- **Analytical derivatives** for accurate force calculations
- **Early exit optimization** for flat-bottom mode
- Performance: ~0.5-1.0 ms per evaluation

### 2. Flexible Atom Selection
- **All atoms**: Use entire structure
- **Index list**: Specify atom indices directly
- **PDB-based selection**: Advanced selection criteria using residue numbers, names, distances, etc.

### 3. Reference Structure Options
- **PDB file**: Read reference from external PDB file
- **Explicit coordinates**: Provide coordinates directly in input
- **Initial coordinates**: Use system's initial coordinates as reference

### 4. Restraint Modes
- **Harmonic**: Standard harmonic restraint `E = 0.5 * k * (RMSD - target)²`
- **Flat-bottom**: One-sided restraint (only applies force when RMSD > target)

### 5. Time-Varying Parameters
- **Target RMSD**: Gradually change target RMSD over simulation
- **Force constants**: Adjust restraint strength over time
- **Interpolation**: Smooth or discrete parameter changes

## Basic Usage

### Simple RMSD Restraint
```yaml
restraints:
  maintain_structure:
    type: rmsd
    target_rmsd: 0.5          # Target RMSD in Angstroms
    force_constant: 2.0       # Force constant
    mode: harmonic            # harmonic or flat_bottom
    atoms: all               # Apply to all atoms
```

### Using PDB Reference File
```yaml
restraints:
  preserve_fold:
    type: rmsd
    reference_file: reference.pdb
    target_rmsd: 0.3
    force_constant: 5.0
    mode: flat_bottom
    atoms: all
```

### Atom Selection Examples
```yaml
# Select specific atoms by index
restraints:
  backbone_rmsd:
    type: rmsd
    target_rmsd: 0.2
    force_constant: 10.0
    atoms: [0, 2, 4, 6, 8]    # Specific atom indices

# Advanced PDB-based selection
restraints:
  active_site_rmsd:
    type: rmsd
    reference_file: crystal_structure.pdb
    target_rmsd: 0.1
    force_constant: 20.0
    atom_selection:
      residue_numbers: [25, 26, 27, 28]  # Specific residues
      atom_names: ["CA", "CB", "CG"]     # Backbone atoms only
      within_distance:                   # Atoms within distance of reference
        reference_atoms: [125]
        distance: 5.0
```

## Time-Varying Restraints

### Gradual Structural Change
```yaml
restraints:
  controlled_unfolding:
    type: rmsd
    reference_file: folded_state.pdb
    target_rmsd: [0.2, 0.5, 1.0, 2.0, 3.0]    # Gradually increase allowed deviation
    force_constant: [20.0, 10.0, 5.0, 2.0, 1.0] # Decrease restraint strength
    mode: harmonic
    update_steps: 2000                         # Change every 2000 steps
    interpolation_mode: interpolate            # Smooth transitions
    atoms: all
```

### Steered MD with RMSD
```yaml
restraints:
  steered_folding:
    type: rmsd
    reference_file: target_structure.pdb
    target_rmsd: [3.0, 2.0, 1.0, 0.5, 0.2]    # Gradually decrease to fold
    force_constant: 5.0
    mode: flat_bottom                          # Only apply force when too far
    update_steps: 1000
    interpolation_mode: discrete               # Step changes
    atom_selection:
      residue_numbers: [10, 11, 12, 13, 14]   # Focus on specific region
```

## Advanced Features

### Multiple RMSD Restraints
```yaml
restraints:
  # Global fold preservation
  global_structure:
    type: rmsd
    target_rmsd: 1.0
    force_constant: 1.0
    atoms: all
    
  # Local active site preservation
  active_site:
    type: rmsd
    reference_file: active_site_reference.pdb
    target_rmsd: 0.2
    force_constant: 10.0
    atom_selection:
      residue_numbers: [25, 26, 27]
      atom_names: ["CA", "CB", "CG", "CD"]
    
  # Secondary structure preservation
  helix_region:
    type: rmsd
    target_rmsd: 0.3
    force_constant: 5.0
    atoms: [100, 101, 102, 103, 104, 105]  # Helix backbone atoms
```

### Combination with Other Restraints
```yaml
restraints:
  # RMSD for overall structure
  global_rmsd:
    type: rmsd
    target_rmsd: 0.8
    force_constant: 2.0
    
  # Distance restraint for specific interaction
  key_distance:
    type: distance
    atoms: [25, 87]
    target_distance: 3.5
    force_constant: 10.0
    
  # Spherical boundary to prevent drift
  boundary:
    type: spherical_boundary
    center: auto
    radius: 25.0
    force_constant: 1.0
```

## Performance Considerations

### System Size Scaling
- **Small systems** (< 100 atoms): ~0.3 ms per evaluation
- **Medium systems** (100-1000 atoms): ~0.5-1.0 ms per evaluation  
- **Large systems** (> 1000 atoms): ~1-5 ms per evaluation

### Optimization Tips
1. **Use atom selection** to reduce computational cost for large systems
2. **Flat-bottom mode** can be more efficient when RMSD is below target
3. **JIT compilation** provides automatic optimization after first few calls
4. **Time-varying restraints** add minimal overhead

### Memory Usage
- Reference coordinates stored once per restraint
- Minimal additional memory overhead
- Vectorized operations reduce memory allocations

## Technical Details

### RMSD Calculation
The RMSD between current coordinates **r** and reference coordinates **r₀** is:

```
RMSD = √(1/N Σᵢ |rᵢ - r₀ᵢ|²)
```

### Force Calculation
For harmonic restraints, the force on atom *i* is:

```
Fᵢ = -k * (RMSD - target) * (1/RMSD) * (1/N) * (rᵢ - r₀ᵢ)
```

Where:
- **k** = force constant
- **N** = number of selected atoms
- **target** = target RMSD value

### Modes
- **Harmonic**: Always applies force proportional to deviation
- **Flat-bottom**: Only applies force when RMSD > target (one-sided)

## Error Handling

### Common Issues
1. **Missing reference**: Must provide `reference_file`, `reference_coordinates`, or have `initial_coordinates`
2. **Mismatched sizes**: Reference and current coordinates must have same number of atoms
3. **Invalid atom selection**: Check atom indices and selection criteria
4. **PDB parsing errors**: Ensure PDB file is properly formatted

### Debugging Tips
- Use `debug: true` in restraint definition for detailed output
- Check RMSD values manually with test configurations
- Verify atom selection using simple index lists first
- Test with small systems before scaling up

## Examples and Use Cases

### 1. Protein Folding Studies
Maintain secondary structure while allowing tertiary changes:
```yaml
restraints:
  secondary_structure:
    type: rmsd
    target_rmsd: 0.5
    force_constant: 5.0
    atom_selection:
      atom_names: ["CA", "C", "N", "O"]  # Backbone only
```

### 2. Ligand Binding Simulations
Preserve binding site geometry:
```yaml
restraints:
  binding_site:
    type: rmsd
    reference_file: apo_structure.pdb
    target_rmsd: 0.3
    force_constant: 10.0
    atom_selection:
      within_distance:
        reference_atoms: [ligand_center_atom]
        distance: 8.0
```

### 3. Conformational Sampling
Allow gradual exploration around reference state:
```yaml
restraints:
  flexible_sampling:
    type: rmsd
    target_rmsd: [0.2, 0.5, 1.0, 0.5, 0.2]  # Expand then contract
    force_constant: 3.0
    mode: flat_bottom
    update_steps: 2000
```

The RMSD restraint provides a powerful and flexible tool for controlling structural similarity during MD simulations while maintaining high performance through optimized implementations.