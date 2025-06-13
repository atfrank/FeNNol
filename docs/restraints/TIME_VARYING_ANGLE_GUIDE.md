# Time-Varying Target Angle for Backside Attack Restraints

This guide explains how to use time-varying target angles in backside attack restraints to help achieve better convergence in SN2 reaction simulations.

## Overview

The time-varying target angle feature allows you to gradually change the target angle during the simulation. This is particularly useful when:
- The initial geometry is far from the desired 180° linear arrangement
- Direct restraints to 180° cause instability or poor convergence
- You want to guide the system through intermediate states

## Basic Usage

To use a time-varying target angle, provide a list of angles instead of a single value:

```
restraints {
  sn2_adaptive {
    type = backside_attack
    nucleophile = 52
    carbon = 245
    leaving_group = 249
    
    # Gradually change from 140° to 180°
    target = [140.0, 150.0, 160.0, 170.0, 180.0]  # degrees
    
    # Update the target angle every 200,000 steps
    update_steps = 200000
    
    # Force constant (can be constant or also time-varying)
    angle_force_constant = 0.01
  }
}
```

## Advanced Example with Multiple Time-Varying Parameters

You can combine time-varying target angles with time-varying force constants:

```
restraints {
  sn2_complex {
    type = backside_attack
    nucleophile = 52
    carbon = 245
    leaving_group = 249
    
    # Gradually change target angle
    target = [140.0, 150.0, 160.0, 170.0, 175.0, 180.0]
    
    # Also increase force constant over time
    angle_force_constant = [0.008, 0.01, 0.012, 0.015]
    
    # Update every 200,000 steps
    update_steps = 200000
    
    # Use flat-bottom restraint for smoother dynamics
    style = flat_bottom
    tolerance = 0.2  # ~11.5 degrees
    
    # Prevent incorrect Nu-LG-C arrangement
    prevent_inversion = true
    inversion_penalty_factor = 1.0
  }
}
```

## Complete Example for SN2 Reaction

Here's a complete restraint setup combining time-varying angles with distance restraints:

```
restraints {
  # Main backside attack restraint with adaptive angle
  backside_attack_restraint {
    type = backside_attack
    nucleophile = 52
    carbon = 245
    leaving_group = 249
    
    # Start at 140° and gradually reach 180°
    target = [140.0, 150.0, 160.0, 170.0, 175.0, 180.0]
    angle_force_constant = [0.008, 0.01, 0.012]
    update_steps = 200000
    
    style = flat_bottom
    tolerance = 0.2
    
    # Also control Nu-C distance
    target_distance = 5.0
    distance_force_constant = 0.01
    
    prevent_inversion = true
  }
  
  # Ensure Nu-C distance < Nu-LG distance
  maintain_order1 {
    type = distance
    atom1 = 52   # nucleophile
    atom2 = 245  # carbon
    target = 6.0
    force_constant = 0.02
    style = flat_bottom
    tolerance = 0.2
  }
  
  maintain_order2 {
    type = distance
    atom1 = 52   # nucleophile
    atom2 = 249  # leaving group
    target = 7.0  # larger than Nu-C target
    force_constant = 0.02
    style = flat_bottom
    tolerance = 0.2
  }
}
```

## How It Works

1. **Linear Interpolation**: The target angle is linearly interpolated between the values in your list based on the current simulation step.

2. **Update Schedule**: With `update_steps = 200000` and 5 target angles:
   - Steps 0-200,000: Interpolate between angle 1 and 2
   - Steps 200,000-400,000: Interpolate between angle 2 and 3
   - And so on...

3. **Smooth Transitions**: The interpolation ensures smooth transitions between target angles, avoiding sudden jumps that could destabilize the simulation.

## Best Practices

1. **Start Conservative**: Begin with angles close to your initial geometry and gradually approach 180°.

2. **Use Flat-Bottom Style**: This provides more flexibility and prevents excessive forces when the angle deviates from the target.

3. **Combine with Distance Restraints**: Use additional distance restraints to maintain the correct Nu-C-LG ordering.

4. **Monitor Progress**: Use collective variables to track the actual angle throughout the simulation:
   ```
   colvars {
     attack_angle {
       type = angle
       atom1 = 52   # nucleophile
       atom2 = 245  # carbon
       atom3 = 249  # leaving group
     }
   }
   ```

5. **Adjust Update Steps**: Choose `update_steps` based on your system's relaxation time. Faster changes may cause instability, while slower changes may waste computational time.

## Troubleshooting

- **If angles don't reach target**: Increase the force constant or extend the simulation time
- **If simulation becomes unstable**: Reduce force constants or increase `update_steps` for slower transitions
- **If wrong arrangement forms**: Ensure `prevent_inversion = true` and adjust `inversion_penalty_factor`

## Technical Details

- Angles can be specified in degrees (values > 2π) or radians
- The restraint energy and forces are calculated using the current interpolated target angle
- All standard restraint styles (harmonic, flat_bottom, one_sided) are supported
- The feature integrates seamlessly with existing distance and force constant time-varying options