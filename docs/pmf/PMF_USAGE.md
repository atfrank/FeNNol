# PMF Estimation Usage Guide

## Overview
The PMF (Potential of Mean Force) estimation feature allows calculation of free energy profiles along reaction coordinates using umbrella sampling with discrete windows.

## Requirements
- `interpolation_mode = discrete` (required)
- `estimate_pmf = yes` in restraint definition
- `restraint_debug yes` to see progress information

## Example Configuration

```fnl
restraints {
  my_pmf_restraint {
    type = backside_attack
    nucleophile = 0
    carbon = 1
    leaving_group = 5
    
    # Fixed angle
    target = 180.0
    angle_force_constant = 20.0
    
    # Time-varying distance for PMF
    target_distance = 3.5 3.2 2.9 2.6 2.3 2.0
    distance_force_constant = 15.0
    
    # PMF-specific settings
    interpolation_mode = discrete    # REQUIRED
    estimate_pmf = yes              # Enable PMF
    equilibration_ratio = 0.3       # 30% equilibration
    update_steps = 3000             # Steps per window
    
    style = harmonic
  }
}

# Enable debug output to see progress
restraint_debug yes
```

## PMF Progress Information

With `restraint_debug yes`, you'll see:

1. **Window Entry**:
```
# PMF [my_pmf_restraint] === ENTERING WINDOW 1/6 ===
#   Target distance: 3.500
#   Equilibration: 900 steps, Sampling: 2100 steps
```

2. **Equilibration Progress**:
```
# PMF [my_pmf_restraint] Window 1: EQUILIBRATING 50.0% | distance=3.512
```

3. **Sampling Progress** (every 500 steps):
```
# PMF [my_pmf_restraint] Window 1: SAMPLING 23.8% | Samples=500 | Current=3.498 | Mean=3.501±0.015
```

4. **Window Completion**:
```
# PMF [my_pmf_restraint] === WINDOW 1 COMPLETE ===
#   Target: 3.500
#   Samples collected: 2100
#   Mean ± StdDev: 3.502 ± 0.016
#   Range: [3.456, 3.548]
```

## Accessing PMF Results

The PMF functions need to be called explicitly in your simulation script:

### During Simulation
```python
from fennol.md.restraints import print_pmf_status

# Call periodically (e.g., every nsummary steps)
print_pmf_status()
```

### At Simulation End
```python
from fennol.md.restraints import print_pmf_final_summary, write_pmf_output

# Print final summary to console
print_pmf_final_summary(temperature=300.0)

# Write detailed results to file
write_pmf_output(filename="pmf_output.dat", temperature=300.0)
```

## Output File Format

The `pmf_output.dat` file contains:
- Window centers
- PMF values in kcal/mol
- Sampling statistics
- Calculated free energy profile

## Important Notes

1. The PMF calculation assumes equilibrium sampling in each window
2. Ensure adequate equilibration time (20-30% recommended)
3. Check convergence by monitoring mean and standard deviation
4. The global step counter is used for consistency
5. PMF values are relative (minimum set to 0)

## Troubleshooting

If you don't see PMF output:
1. Verify `restraint_debug yes` is set
2. Check that `interpolation_mode = discrete`
3. Ensure `estimate_pmf = yes` for the restraint
4. Confirm the restraint has time-varying parameters
5. Make sure to call the PMF output functions

## Integration with MD Scripts

For automated PMF output, modify your MD script to call the PMF functions:

```python
# In your MD loop
if step % nsummary == 0:
    print_pmf_status()

# After MD completes
print_pmf_final_summary(temperature)
write_pmf_output("pmf_results.dat", temperature)
```