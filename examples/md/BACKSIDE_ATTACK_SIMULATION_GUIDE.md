# Backside Attack Simulation Guide

## Overview

The backside attack restraint is designed to facilitate nucleophilic substitution (SN2) reactions by enforcing the characteristic linear approach geometry. This guide explains what to expect when simulating a backside attack, especially when the nucleophile and carbon are initially far apart.

## Angle Units

The `target` angle for backside attack restraints can be specified in either **degrees** or **radians**:

- **Degrees**: Use values > 2π (e.g., `target = 180.0` for a linear approach)
- **Radians**: Use values ≤ 2π (e.g., `target = 3.14159` for π radians = 180°)

The system automatically detects the unit based on the magnitude of the value. This makes it more intuitive to specify common angles like 180°, 170°, etc.

## Inversion Prevention (New Feature)

The backside attack restraint now includes automatic **inversion prevention** to avoid the incorrect Nu-LG-C arrangement. This feature:

- **Detects** when Nu-C distance > Nu-LG distance (wrong ordering)
- **Applies penalty forces** to correct the arrangement
- **Is enabled by default** with `prevent_inversion = true`

### Parameters:

```
restraints {
  sn2_safe {
    type = backside_attack
    nucleophile = 5
    carbon = 0
    leaving_group = 4
    
    target = 180.0
    angle_force_constant = 10.0
    
    # Inversion prevention parameters
    prevent_inversion = true      # Default: true
    inversion_penalty_factor = 2.0  # Default: 2.0 (multiplies angle_force_constant)
  }
}
```

### How It Works:

1. **Monitors distances**: Continuously checks Nu-C vs Nu-LG distances
2. **Applies penalty**: When Nu-C > Nu-LG (indicating wrong order), applies:
   - Forces pushing Nu away from LG
   - Forces pulling Nu toward C
3. **Penalty strength**: `penalty_force = angle_force_constant × inversion_penalty_factor × violation`

### Best Practices:

1. **Always use with distance restraints** for maximum effectiveness:
   ```
   target_distance = 2.5
   distance_force_constant = 10.0
   ```

2. **Increase penalty for stubborn systems**:
   ```
   inversion_penalty_factor = 3.0  # Stronger correction
   ```

3. **Monitor with colvars** to verify correct geometry:
   ```
   colvars {
     nu_c_distance { type = distance; atom1 = 5; atom2 = 0 }
     nu_lg_distance { type = distance; atom1 = 5; atom2 = 4 }
   }
   ```

## Time-Varying Force Constants

The backside attack restraint **fully supports time-varying force constants** and can be configured as either **angle-only** or **composite** (angle + distance).

#### Angle-Only Restraint:
```
restraints {
  sn2_angle_only {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    leaving_group = 12
    target = 180.0  # Target angle in degrees (can also use 3.14159 for radians)
    
    # Gradually increase angle force constant
    angle_force_constant = [0.1, 2.0, 10.0, 20.0]  # kcal/mol/rad²
    update_steps = 50000  # Change every 50,000 steps
    style = harmonic
  }
}
```

#### Composite Restraint (Recommended):
```
restraints {
  sn2_composite {
    type = backside_attack
    nucleophile = 15
    carbon = 5
    leaving_group = 12
    
    # Angle component
    target = 180.0  # Degrees (values > 2π are treated as degrees)
    angle_force_constant = [0.1, 2.0, 10.0, 20.0]
    
    # Distance component  
    target_distance = 2.5  # Angstroms
    distance_force_constant = [5.0, 15.0, 10.0, 2.0]
    
    update_steps = 50000
    style = harmonic
  }
}
```

## Simulation Dynamics: What to Expect

### Phase 1: Initial Approach (Far Separation)
When the nucleophile and carbon are initially very far apart:

#### With Angle-Only Restraint:
**What happens:**
- The backside attack restraint applies **no force** initially because the angle is undefined or very small
- The nucleophile will undergo random thermal motion
- **Key limitation**: The angle restraint alone cannot bring distant atoms together

**Solution:** Use separate distance restraint or composite restraint.

#### With Composite Restraint (Recommended):
**What happens:**
- The **distance component** immediately starts pulling the nucleophile toward the carbon
- The **angle component** becomes active once they're within reasonable range (~4-5 Å)
- This provides smooth approach dynamics without requiring separate restraints

**Example composite restraint:**
```
sn2_composite {
  type = backside_attack
  nucleophile = 15
  carbon = 5
  target = 3.14159
  angle_force_constant = [0.1, 2.0, 10.0]    # Gradual angle control
  target_distance = 2.8
  distance_force_constant = [8.0, 12.0, 5.0] # Strong initial pull, then moderate
  update_steps = 50000
  style = harmonic
}
```

### Phase 2: Angle Alignment (Medium Range)
Once the nucleophile approaches within ~4-5 Å of the carbon:

**What happens:**
- The backside attack restraint becomes active
- Forces will orient the nucleophile to the opposite side of the leaving group
- The nucleophile will start to "feel" the linear geometry constraint
- You'll see the attack angle gradually approach 180°

### Phase 3: Close Approach and Reaction
As the nucleophile gets closer (< 3 Å):

**What happens:**
- Strong angular forces maintain the backside approach
- The reaction can proceed along the SN2 pathway
- The leaving group may start to dissociate
- Bond breaking/forming can occur

## Factors Controlling Reaction Speed

### 1. **Force Constant Magnitude**
```
force_constant = 10.0   # Moderate speed
force_constant = 50.0   # Faster approach
force_constant = 1.0    # Slower, more natural approach
```

**Effect:** Higher force constants = faster approach, but risk of simulation instability

### 2. **Time-Varying Profile**
```
# Gradual approach (recommended)
force_constant = [0.1, 1.0, 5.0, 15.0]
update_steps = 25000

# Rapid approach
force_constant = [0.5, 10.0, 30.0]
update_steps = 10000
```

**Effect:** The rate of force constant increase controls how quickly the restraint "takes over"

### 3. **Temperature**
```
temperature = 300.0  # Room temperature - moderate kinetics
temperature = 500.0  # Higher temperature - faster dynamics
temperature = 200.0  # Lower temperature - slower approach
```

**Effect:** Higher temperatures provide more thermal energy to overcome barriers

### 4. **Friction (Langevin Thermostat)**
```
gamma[THz] = 0.1   # Low friction - faster dynamics
gamma[THz] = 1.0   # Moderate friction (typical)
gamma[THz] = 10.0  # High friction - slower, more controlled
```

**Effect:** Lower friction allows faster motion; higher friction provides more control

### 5. **Restraint Style**
```
style = harmonic      # Constant force proportional to deviation
style = flat_bottom   # No force within tolerance, then harmonic
tolerance = 0.1       # Small tolerance = tight control
tolerance = 0.3       # Larger tolerance = more flexibility
```

## Recommended Strategy for Long-Range Approach

### Multi-Stage Protocol:

1. **Initial Approach (0-100,000 steps)**:
   ```
   # Weak distance restraint to bring nucleophile closer
   force_constant = [5.0, 10.0]
   target = 4.0  # Angstroms
   ```

2. **Angle Alignment (100,000-250,000 steps)**:
   ```
   # Moderate backside attack restraint
   force_constant = [0.1, 2.0, 8.0]
   target = 3.14159
   ```

3. **Final Approach (250,000+ steps)**:
   ```
   # Strong restraint for reaction
   force_constant = [8.0, 20.0]
   target = 3.14159
   ```

## Expected Observables

### Attack Angle Evolution:
- **Initial**: Random, small angles (< 90°)
- **Early approach**: Gradual increase toward 150-160°
- **Close approach**: Approaches 180° ± 10°
- **Reaction**: Maintains ~180° as bonds break/form

### Distance Evolution:
- **Nucleophile-Carbon**: Decreases from initial separation to ~2-2.5 Å
- **Carbon-Leaving Group**: Initially stable, then increases during reaction
- **Reaction Coordinate**: Becomes negative as reaction proceeds

### Energy Profile:
- **Restraint Energy**: Starts high, decreases as geometry improves
- **Total Energy**: May increase initially due to restraint forces
- **Kinetic Energy**: Fluctuates with temperature control

## Troubleshooting

### Problem: Nucleophile doesn't approach
**Solution**: Add distance restraint or increase force constant

### Problem: Simulation becomes unstable
**Solution**: Reduce force constant, use time-varying profile, or increase friction

### Problem: Wrong approach geometry
**Solution**: Check leaving group assignment, verify target angle (should be π)

### Problem: Reaction too fast/slow
**Solution**: Adjust temperature, force constant profile, or update steps

## Example Timeline for Typical SN2 Reaction

- **0-50 ps**: Random motion, nucleophile searches for carbon
- **50-200 ps**: Initial approach, distance decreases to ~4 Å
- **200-500 ps**: Angle alignment, approach angle increases to ~160°
- **500-800 ps**: Close approach, angle reaches ~175-180°
- **800+ ps**: Reaction can occur, bonds break/form

This timeline will vary significantly based on system size, temperature, force constants, and chemical identity of the reactants.