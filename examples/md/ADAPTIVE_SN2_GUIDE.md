# Adaptive SN2 Restraint Guide

## Overview

The adaptive SN2 restraint (`type = adaptive_sn2`) is a sophisticated restraint system that automatically configures itself to guide SN2 nucleophilic substitution reactions. Unlike standard restraints that require manual parameter tuning, this restraint intelligently adjusts its parameters throughout the simulation to ensure a smooth, physically realistic reaction pathway.

## Key Features

1. **Automatic Parameter Determination**: All force constants and target values are automatically calculated based on the initial molecular geometry and simulation length.

2. **Three-Phase Reaction Path**:
   - **Phase 1 (20%)**: Gentle nucleophile approach while maintaining initial geometry
   - **Phase 2 (50%)**: Gradual angle adjustment to achieve backside attack geometry
   - **Phase 3 (30%)**: Final reaction with strong SN2 enforcement

3. **Inversion Prevention**: Automatically prevents incorrect Nu-LG-C ordering that would lead to unphysical geometries.

4. **Smooth Transitions**: All parameters change gradually to avoid sudden forces that could distort the system.

## Basic Usage

```fnl
restraints {
  sn2_auto {
    type = adaptive_sn2
    nucleophile = 5       # Index of attacking atom
    carbon = 0            # Index of carbon center
    simulation_steps = 100000  # Total simulation steps
  }
}
```

## Parameters

### Required Parameters
- `nucleophile`: Atom index of the nucleophile (attacking species)
- `carbon`: Atom index of the carbon being attacked
- `simulation_steps`: Total number of simulation steps (must match `nsteps`)

### Optional Parameters
- `leaving_group`: Atom index of the leaving group (auto-detected if not specified)

## How It Works

### Phase 1: Initial Approach (0-20% of simulation)
- Maintains the initial Nu-C-LG angle
- Gradually reduces Nu-C distance from initial to ~4.0 Å
- Uses soft constraints (fc ≈ 0.02-0.05 kcal/mol/Å²)

### Phase 2: Angle Adjustment (20-70% of simulation)
- Smoothly transitions the angle from initial to 180°
- Continues Nu-C approach from 4.0 to 2.5 Å
- Gradually increases force constants
- Begins C-LG bond elongation

### Phase 3: Reaction Completion (70-100% of simulation)
- Enforces linear Nu-C-LG geometry (180° ± 5.7°)
- Final Nu-C approach from 2.5 to 1.7 Å
- Strong force constants to drive reaction
- Active C-LG bond breaking assistance

## Automatic Features

### Leaving Group Detection
If `leaving_group` is not specified, the restraint identifies it as the atom:
- Bonded to the carbon (within 2.0 Å)
- Furthest from the nucleophile

### Inversion Prevention
The restraint automatically prevents the incorrect Nu-LG-C arrangement by:
- Monitoring Nu-C and Nu-LG distances
- Applying corrective forces when Nu-LG < Nu-C + 0.5 Å
- Force strength increases in later phases

### Adaptive Force Constants
Force constants are automatically scaled based on:
- Current phase and progress within phase
- Initial molecular geometry
- Distance between reacting atoms

## Example: CH₃Cl + OH⁻ → CH₃OH + Cl⁻

```fnl
# Minimal configuration - everything auto-configured
restraints {
  sn2_reaction {
    type = adaptive_sn2
    nucleophile = 5       # OH- oxygen
    carbon = 0            # Methyl carbon
    simulation_steps = 100000
  }
}
```

## Monitoring Progress

Enable restraint debugging to see detailed information:

```fnl
restraint_debug = true
```

This will output:
- Current phase and progress
- Current vs. target angles and distances
- Applied force constants
- Energy contributions from each component

## Best Practices

1. **Match simulation steps**: Ensure `simulation_steps` matches your `nsteps` parameter
2. **Initial geometry**: Start with reasonable initial geometry (Nu-C distance < 10 Å)
3. **Temperature**: Use moderate temperatures (200-400 K) for smooth trajectories
4. **Time step**: Use conservative time steps (0.5-1.0 fs) during reaction
5. **Monitoring**: Always include relevant collective variables to track progress

## Advantages Over Manual Restraints

1. **No parameter tuning**: Automatically determines appropriate force constants
2. **Smooth reaction path**: Three-phase approach prevents sudden geometry changes
3. **Robust**: Inversion prevention ensures correct reaction mechanism
4. **Adaptive**: Adjusts to different molecular systems without modification

## Troubleshooting

### Reaction too fast/slow
- Adjust total `simulation_steps` - the restraint will automatically rescale

### System distortion
- Check initial geometry is reasonable
- Reduce simulation temperature
- Enable `restraint_debug` to identify problematic phases

### Wrong leaving group detected
- Explicitly specify `leaving_group` parameter
- Check atom numbering in your input file