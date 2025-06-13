# Backside Attack Restraint

## Overview

The backside attack restraint facilitates nucleophilic substitution (SN2) reactions by enforcing the characteristic linear approach geometry. It can be configured as either an angle-only restraint or a composite restraint (angle + distance).

## Key Features

- **Linear Geometry Enforcement**: Ensures nucleophile approaches from the opposite side of the leaving group
- **Automatic Leaving Group Detection**: Can identify the most appropriate leaving group atom
- **Composite Restraint Support**: Optional distance component to pull nucleophile toward carbon
- **Separate Force Constants**: Independent control of angle and distance components
- **Time-Varying Profiles**: Gradual force application for stable simulations

## Documentation

For detailed information, please see the following guides:

- [Backside Attack Enhanced Features Guide](backside_attack/BACKSIDE_ATTACK_ENHANCED_FEATURES.md): Full feature documentation
- [Backside Attack Simulation Guide](backside_attack/BACKSIDE_ATTACK_SIMULATION_GUIDE.md): Explanation of expected simulation behavior
- [Adaptive SN2 Guide](backside_attack/ADAPTIVE_SN2_GUIDE.md): Advanced use cases for adaptive SN2 simulations

## Usage Examples

### Basic Angle-Only Restraint

```
restraints {
  sn2_restraint {
    type = backside_attack
    nucleophile = 15      # Index of attacking atom (e.g., OH- oxygen)
    carbon = 5            # Index of carbon being attacked
    leaving_group = 12    # Index of leaving group (optional - can be auto-detected)
    target = 3.14159      # Target angle in radians (π = 180°)
    angle_force_constant = 10.0
    style = harmonic
  }
}
```

### Composite Restraint (Angle + Distance)

```
restraints {
  sn2_composite {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    
    # Angle component
    target = 3.14159                # Target angle: 180°
    angle_force_constant = 10.0     # Force constant for angle
    
    # Distance component
    target_distance = 2.5           # Target nucleophile-carbon distance (Å)
    distance_force_constant = 8.0   # Force constant for distance
    
    style = harmonic
  }
}
```

### Time-Varying Force Constants

```
restraints {
  sn2_advanced {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    
    # Independent time-varying force constants
    target = 3.14159
    angle_force_constant = [0.1, 2.0, 10.0, 20.0]      # Gradual angle increase
    target_distance = 2.8
    distance_force_constant = [15.0, 12.0, 8.0, 2.0]   # Strong initial pull, then relax
    
    update_steps = 25000
    style = harmonic
  }
}
```

## Example Files

Several example input files demonstrating different aspects of the backside attack restraint are provided in `examples/md/backside_attack/`:

- `backside_attack_example.fnl`: Basic restraint setup
- `backside_attack_composite.fnl`: Composite restraint with angle and distance components
- `backside_attack_time_varying.fnl`: Time-varying force constants
- `backside_attack_time_varying_angle.fnl`: Special angle handling with time-varying parameters
- `sn2_time_varying_angle_minimal.fnl`: Minimal example for SN2 reaction

## Implementation Details

The backside attack restraint is implemented in `src/fennol/md/restraints.py` using JAX for automatic differentiation. It builds on the existing restraint framework, extending it with specialized SN2 reaction support.

## Performance Considerations

- The composite restraint (angle + distance) is computationally equivalent to having separate restraints
- Time-varying force constants have minimal performance impact
- Automatic leaving group detection adds a small computational cost on each step