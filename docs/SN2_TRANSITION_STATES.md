# SN2 Transition State Search in FeNNol

FeNNol includes specialized functionality for finding transition states in SN2 (Substitution Nucleophilic bimolecular) reactions. This bespoke method leverages the well-defined reaction coordinate of SN2 reactions to efficiently locate the transition state.

## Overview

SN2 reactions follow a characteristic mechanism where a nucleophile attacks a carbon center while a leaving group departs, all occurring simultaneously through a concerted mechanism. The transition state features:

- Backside attack by the nucleophile
- Partial bonds to both nucleophile and leaving group
- Inversion of stereochemistry at the carbon center
- Linear arrangement of Nu-C-LG atoms

## SN2 Reaction Coordinate

The SN2 method uses a specific reaction coordinate definition:

```
RC = d(Nu-C) - d(C-LG)
```

Where:
- `d(Nu-C)` is the distance between nucleophile and carbon
- `d(C-LG)` is the distance between carbon and leaving group

At the transition state, this reaction coordinate should be approximately zero, indicating equal partial bonding to both Nu and LG.

## Usage

### Basic SN2 Input File

```fnl
# SN2 transition state optimization
transition_state    = True
ts_only            = True
ts_method          = sn2

# System definition
coordinates        = sn2_initial_guess.xyz
model_type         = ani2x

# SN2-specific parameters (1-based atom indices)
sn2_nu_index       = 1    # Nucleophile atom index
sn2_c_index        = 2    # Carbon center index  
sn2_lg_index       = 6    # Leaving group index

# Optimization parameters
min_force_tolerance = 1e-3
min_max_iterations  = 200
```

### Required Parameters

#### Core SN2 Parameters
- `sn2_nu_index`: 1-based index of the nucleophile atom
- `sn2_c_index`: 1-based index of the carbon center
- `sn2_lg_index`: 1-based index of the leaving group atom

#### Optional SN2 Parameters
- `sn2_target_nu_c_distance`: Target Nu-C distance (default: 2.0 Å)
- `sn2_target_c_lg_distance`: Target C-LG distance (default: 2.0 Å)
- `sn2_constraint_strength`: Constraint force strength (default: 0.1)
- `sn2_reaction_coordinate_weight`: RC following weight (default: 1.0)
- `sn2_initial_step`: Initial step size (default: 0.05 Å)

## Algorithm Details

### 1. Reaction Coordinate Calculation
The method continuously monitors the reaction coordinate:
```
RC = |r_Nu - r_C| - |r_C - r_LG|
```

### 2. Constraint Forces
Geometric constraints prevent the system from exploring unrealistic configurations:
- Nu-C distance constraint: Prevents nucleophile from moving too far
- C-LG distance constraint: Maintains reasonable leaving group distance

### 3. Optimization Strategy
The algorithm follows a hybrid approach:
- **Far from TS** (|RC| > 0.1): Follow reaction coordinate toward RC = 0
- **Near TS** (|RC| ≤ 0.1): Minimize forces perpendicular to RC

### 4. Convergence Criteria
Convergence requires both:
- Force magnitude below tolerance
- Reaction coordinate close to zero (|RC| < 0.1)

## Example Systems

### Chloride + Methyl Bromide
```
Cl⁻ + CH₃Br → ClCH₃ + Br⁻
```

Initial TS guess coordinates:
```xyz
8
SN2 reaction: Cl- + CH3Br -> ClCH3 + Br-
Cl  -2.5000    0.0000    0.0000
C    0.0000    0.0000    0.0000  
H    0.0000    1.0900    0.0000
H    0.9439   -0.5450    0.0000
H   -0.9439   -0.5450    0.0000
Br   2.5000    0.0000    0.0000
H    0.0000    0.0000    1.5000
H    0.0000    0.0000   -1.5000
```

Input file parameters:
```fnl
sn2_nu_index = 1  # Cl⁻
sn2_c_index  = 2  # C
sn2_lg_index = 6  # Br⁻
```

### Hydroxide + Alkyl Halide
```
OH⁻ + R-X → R-OH + X⁻
```

The method works for any SN2 system by simply adjusting the atom indices.

## Output Information

The SN2 optimizer provides additional output:
- Current reaction coordinate value
- Nu-C and C-LG distances
- Constraint force contributions
- Progress toward optimal geometry

Example output:
```
# SN2 TS optimization completed in 45 steps
# Converged: True
# Final reaction coordinate: 0.0023
# Final Nu-C distance: 2.156 Å
# Final C-LG distance: 2.153 Å
```

## Best Practices

### 1. Initial Guess Preparation
- Start with approximate TS geometry
- Place nucleophile and leaving group equidistant from carbon
- Ensure linear or near-linear Nu-C-LG arrangement

### 2. Parameter Tuning
- **Large systems**: Reduce `sn2_initial_step` and increase `min_max_iterations`
- **Flexible systems**: Increase `sn2_constraint_strength`
- **Tight binding**: Adjust target distances based on chemical intuition

### 3. Convergence Issues
If optimization doesn't converge:
1. Check atom indices are correct
2. Improve initial guess geometry
3. Adjust constraint parameters
4. Try different force tolerance

### 4. Validation
Always verify the final structure:
- Visual inspection of geometry
- Check that Nu-C-LG atoms are approximately linear
- Confirm reasonable bond distances
- Perform frequency calculation if available

## Limitations

1. **Concerted mechanism only**: Only works for concerted SN2 reactions
2. **Single TS**: Assumes one transition state between reactants and products
3. **Gas phase**: May need adjustment for solvated systems
4. **Bond lengths**: Target distances may need adjustment for different atom types

## Troubleshooting

### Common Issues

**Problem**: Optimization diverges
**Solution**: Reduce `sn2_initial_step` or improve initial guess

**Problem**: Wrong atom indices error
**Solution**: Check that indices correspond to correct atoms (1-based)

**Problem**: High final RC value
**Solution**: Increase `min_max_iterations` or adjust constraint strength

**Problem**: Forces don't converge
**Solution**: Relax `min_force_tolerance` or check for system instabilities

### Debugging Tips

1. Monitor RC value during optimization
2. Check that distances remain reasonable
3. Visualize trajectory to understand optimization path
4. Compare with known experimental or theoretical TS structures

## Integration with FeNNol Workflow

The SN2 method integrates seamlessly with FeNNol's molecular dynamics framework:

```bash
# Run SN2 TS optimization
fennol_ts sn2_input.fnl

# Or use as part of MD workflow
fennol_md sn2_with_md.fnl
```

The optimized TS structure can be used for:
- Reaction rate calculations
- Intrinsic reaction coordinate (IRC) calculations  
- Kinetic isotope effect predictions
- Mechanistic studies