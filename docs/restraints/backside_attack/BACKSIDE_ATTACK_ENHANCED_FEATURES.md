# Enhanced Backside Attack Restraint Features

## Overview

The backside attack restraint has been enhanced to support **composite restraints** with separate force constants for angle and distance components. This provides much better control over SN2 reaction simulations, especially when the nucleophile and carbon start far apart.

## Key Enhancements

### 1. **Composite Restraint Support**
The restraint can now control both angle and distance simultaneously:

```
restraints {
  sn2_composite {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    leaving_group = 12
    
    # Angle component (always present)
    target = 3.14159                # 180° backside approach
    angle_force_constant = 10.0
    
    # Distance component (optional)
    target_distance = 2.5           # Nucleophile-carbon distance in Å
    distance_force_constant = 8.0
    
    style = harmonic
  }
}
```

### 2. **Separate Time-Varying Force Constants**
Each component can have independent time-varying profiles:

```
restraints {
  sn2_advanced {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    
    # Angle: gradual increase
    target = 3.14159
    angle_force_constant = [0.1, 2.0, 10.0, 20.0]
    
    # Distance: strong initial pull, then relax
    target_distance = 2.8
    distance_force_constant = [15.0, 12.0, 8.0, 3.0]
    
    update_steps = 25000
    style = harmonic
  }
}
```

### 3. **Backward Compatibility**
Existing input files continue to work unchanged:

```
# Old syntax still works
sn2_legacy {
  type = backside_attack
  nucleophile = 15
  carbon = 5
  target = 3.14159
  force_constant = 10.0  # Applied to angle component
  style = harmonic
}
```

## Parameter Reference

### Required Parameters:
- `nucleophile`: Index of attacking atom
- `carbon`: Index of carbon being attacked

### Optional Parameters:
- `leaving_group`: Index of leaving group (auto-detected if omitted)
- `target`: Target angle in radians (default: π)
- `target_distance`: Target nucleophile-carbon distance in Å (enables distance component)

### Force Constants:
- `angle_force_constant`: Force constant for angle component (required)
- `distance_force_constant`: Force constant for distance component (required if `target_distance` specified)
- `force_constant`: Legacy parameter, equivalent to `angle_force_constant`

### Time-Varying Support:
- Both force constants can be lists for time-varying behavior
- `update_steps`: Steps between force constant changes
- Independent timing for angle vs distance components

## Advantages of Composite Restraints

### 1. **Better Long-Range Approach**
- **Problem**: Angle-only restraints can't bring distant atoms together
- **Solution**: Distance component provides initial attraction

### 2. **Smoother Reaction Dynamics**
- **Problem**: Sudden strong forces can destabilize simulations
- **Solution**: Gradual force constant changes with independent control

### 3. **Reduced Need for Multiple Restraints**
- **Problem**: Previously needed separate distance + angle restraints
- **Solution**: Single composite restraint handles both aspects

### 4. **Fine-Tuned Control**
- **Problem**: Single force constant affects both approach and geometry
- **Solution**: Separate tuning of distance vs angle components

## Recommended Usage Patterns

### Pattern 1: Far Initial Separation
```
sn2_far_start {
  type = backside_attack
  nucleophile = 15
  carbon = 5
  target = 3.14159
  
  # Strong initial distance pull, moderate angle control
  angle_force_constant = [0.5, 2.0, 8.0]
  target_distance = 3.0
  distance_force_constant = [12.0, 15.0, 8.0]
  
  update_steps = 50000
  style = harmonic
}
```

### Pattern 2: Close Initial Separation
```
sn2_close_start {
  type = backside_attack
  nucleophile = 15
  carbon = 5
  target = 3.14159
  
  # Focus on angle alignment, light distance constraint
  angle_force_constant = [2.0, 10.0, 20.0]
  target_distance = 2.5
  distance_force_constant = [3.0, 5.0, 2.0]
  
  update_steps = 30000
  style = flat_bottom
  tolerance = 0.1
}
```

### Pattern 3: Flexible Distance, Strict Angle
```
sn2_flexible {
  type = backside_attack
  nucleophile = 15
  carbon = 5
  target = 3.14159
  
  # Weak distance constraint, strong angle constraint
  angle_force_constant = 15.0
  target_distance = 2.8
  distance_force_constant = 3.0
  
  style = flat_bottom
  tolerance = 0.15  # Applied to both angle and distance
}
```

## Performance Considerations

### Energy Calculation:
- Composite energy = Angle energy + Distance energy
- Both components use the same restraint function (harmonic, flat_bottom, etc.)
- Computational cost scales linearly with number of components

### Force Calculation:
- Forces from both components are additive
- JAX automatic differentiation handles composite gradients efficiently
- No significant performance penalty vs separate restraints

## Migration Guide

### From Separate Restraints:
```
# Old approach (two separate restraints)
restraints {
  sn2_angle {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    target = 3.14159
    force_constant = 10.0
  }
  
  sn2_distance {
    type = distance
    atom1 = 15
    atom2 = 5
    target = 2.5
    force_constant = 8.0
  }
}

# New approach (single composite restraint)
restraints {
  sn2_composite {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    target = 3.14159
    angle_force_constant = 10.0
    target_distance = 2.5
    distance_force_constant = 8.0
  }
}
```

### Benefits of Migration:
- Simplified input files
- Better time-varying coordination
- Reduced complexity in restraint management
- Equivalent or better performance