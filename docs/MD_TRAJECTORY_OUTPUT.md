# Trajectory Output Configuration in FeNNol MD

## Overview

FeNNol MD simulations now support flexible trajectory output path configuration, allowing you to specify custom file paths and prefixes for trajectory files.

## Configuration Options

### 1. Default Behavior
By default, trajectory files are named based on the input XYZ filename:
```fnl
xyz_input {
  file my_system.xyz
}
# Creates: my_system.traj.xyz (or .arc, .extxyz depending on format)
```

### 2. Using output_prefix
Specify a custom prefix for all output files:
```fnl
output_prefix = /path/to/output/my_simulation
# Creates: /path/to/output/my_simulation.traj.xyz
```

### 3. Using trajectory_file
Specify the complete trajectory file path (extension will be added):
```fnl
trajectory_file = /custom/path/trajectory_output
traj_format = xyz
# Creates: /custom/path/trajectory_output.traj.xyz
```

## Trajectory Formats

The trajectory format is controlled by the `traj_format` parameter:

```fnl
traj_format = xyz     # Creates .traj.xyz file (default for most systems)
traj_format = arc     # Creates .arc file (TINKER format)
traj_format = extxyz  # Creates .traj.extxyz file (extended XYZ with properties)
```

## Examples

### Example 1: Output to specific directory
```fnl
# Save trajectory to results directory
output_prefix = results/sn2_reaction
traj_format = xyz
```

### Example 2: Organize by date
```fnl
# Organize outputs by date
output_prefix = simulations/2024-08-26/run1
```

### Example 3: Full path specification
```fnl
# Specify complete path
trajectory_file = /home/user/project/trajectories/rna_dynamics
```

## Additional Output Files

When using custom paths, other output files follow the same pattern:

- **PIMD beads**: `{output_prefix}_bead001.traj.xyz`, etc.
- **Centroid**: `{output_prefix}_centroid.traj.xyz`
- **Ensemble weights**: `{output_prefix}.ensemble_weights.traj`

## Integration with Multi-Site Restraints

When using multi-site restraints with selectivity tracking, you can also specify the output path:

```fnl
restraints {
  multi_site_rna {
    type = multi_site_backside
    # ... other parameters ...
    selectivity_output = /path/to/analysis/selectivity.dat
  }
}
```

## Best Practices

1. **Use absolute paths** when specifying custom output locations
2. **Create directories first** - FeNNol won't create missing directories
3. **Include descriptive names** in your output prefix for easy identification
4. **Organize by project** - use directory structure to organize simulations

## Backward Compatibility

The new parameters are fully backward compatible:
- If no `output_prefix` or `trajectory_file` is specified, the original behavior is maintained
- Existing scripts will continue to work without modification