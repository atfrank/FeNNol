# Using Restraints in FeNNol MD Simulations

This document explains how to use the restraints feature in FeNNol molecular dynamics simulations. Restraints can be used to apply artificial forces that limit the movement of atoms, maintain specific geometries, or guide conformational changes.

## Types of Restraints

FeNNol supports several types of restraints:

1. **Distance Restraints**: Constrain the distance between two atoms
2. **Angle Restraints**: Constrain the angle formed by three atoms
3. **Dihedral Restraints**: Constrain the dihedral angle formed by four atoms
4. **Backside Attack Restraints**: Guide nucleophilic substitution reactions by enforcing linear approach geometry

Each of these can be implemented with different restraint styles:

- **Harmonic**: Standard harmonic potential, with energy proportional to the square of the deviation from the target value
- **Flat-bottom**: No energy penalty within a tolerance region, harmonic outside of it

## Using Restraints in Input Files

Restraints are defined in a `restraints` section in your input file. Each restraint has a unique name and specific parameters.

### Basic Structure

```
restraints {
  restraint_name {
    type = [distance|angle|dihedral|backside_attack]
    atom1 = [atom_index]
    atom2 = [atom_index]
    ...
    target = [target_value]
    force_constant = [force_constant]
    style = [harmonic|flat_bottom]
    tolerance = [tolerance]  # Only needed for flat_bottom style
  }
  
  another_restraint {
    ...
  }
}
```

### Parameters

For all restraint types:
- `type`: Can be `distance`, `angle`, or `dihedral`
- `target`: Target value (in Angstroms for distances, radians for angles and dihedrals)
- `force_constant`: Force constant (k) for the restraint
- `style`: Either `harmonic` or `flat_bottom`
- `tolerance`: Width of the flat region for `flat_bottom` style (in the same units as `target`)

Additional parameters depend on the restraint type:
- Distance restraints: `atom1`, `atom2`
- Angle restraints: `atom1`, `atom2`, `atom3` (with `atom2` as the vertex)
- Dihedral restraints: `atom1`, `atom2`, `atom3`, `atom4`
- Backside attack restraints: `nucleophile`, `carbon`, `leaving_group` (optional), `target_distance` (optional), `angle_force_constant`, `distance_force_constant` (optional)

### Energy Functions

1. **Harmonic Restraint**:
   ```
   E = 0.5 * k * (value - target)²
   ```

2. **Flat-bottom Restraint**:
   ```
   E = 0.5 * k * (|value - target| - tolerance)²  if |value - target| > tolerance
   E = 0                                         if |value - target| ≤ tolerance
   ```

## Example Applications

### Position Restraints
Restrain an atom to its initial position by using a distance restraint with both indices pointing to the same atom and a target of 0.0.

```
position_restraint {
  type = distance
  atom1 = 10  # Atom to restrain
  atom2 = 10  # Same atom
  target = 0.0
  force_constant = 5.0
  style = flat_bottom
  tolerance = 1.0  # Allow movement within 1 Å
}
```

### Backbone Restraints for Proteins
Maintain secondary structure while allowing sidechains to move:

```
backbone_restraint {
  type = distance
  atom1 = 24  # CA atom
  atom2 = 24  # Same atom
  target = 0.0
  force_constant = 5.0
  style = flat_bottom
  tolerance = 1.0
}
```

### Distance Restraints
Keep two groups within a certain distance:

```
distance_restraint {
  type = distance
  atom1 = 50
  atom2 = 150
  target = 10.0
  force_constant = 2.0
  style = harmonic
}
```

### Dihedral Restraints
Maintain specific conformations in flexible molecules:

```
dihedral_restraint {
  type = dihedral
  atom1 = 1
  atom2 = 4
  atom3 = 7
  atom4 = 10
  target = 3.14  # 180 degrees in radians
  force_constant = 1.0
  style = flat_bottom
  tolerance = 0.17  # ~10 degrees
}
```

### Backside Attack Restraints
Guide nucleophilic substitution (SN2) reactions by enforcing linear approach geometry. This restraint can be either **angle-only** or **composite** (angle + distance).

#### Basic Angle-Only Restraint:
```
sn2_restraint {
  type = backside_attack
  nucleophile = 15      # Index of attacking atom (e.g., OH- oxygen)
  carbon = 5            # Index of carbon being attacked
  leaving_group = 12    # Index of leaving group (optional - can be auto-detected)
  target = 3.14159      # Target angle in radians (π = 180°)
  angle_force_constant = 10.0
  style = harmonic
}
```

#### Composite Restraint (Angle + Distance):
```
sn2_composite {
  type = backside_attack
  nucleophile = 15
  carbon = 5
  leaving_group = 12
  
  # Angle component
  target = 3.14159                # Target angle: 180°
  angle_force_constant = 10.0     # Force constant for angle
  
  # Distance component
  target_distance = 2.5           # Target nucleophile-carbon distance (Å)
  distance_force_constant = 8.0   # Force constant for distance
  
  style = harmonic
}
```

The backside attack restraint enforces a linear nucleophile-carbon-leaving_group geometry (180° angle), which is characteristic of SN2 reactions. If `leaving_group` is not specified, the restraint will automatically detect the atom bonded to the carbon that is furthest from the nucleophile.

**Separate time-varying force constants** are supported for each component:

```
sn2_advanced {
  type = backside_attack
  nucleophile = 15
  carbon = 5
  target = 3.14159
  
  # Independent time-varying force constants
  angle_force_constant = [0.1, 2.0, 10.0, 20.0]      # Angle component
  target_distance = 2.8
  distance_force_constant = [5.0, 15.0, 10.0, 2.0]   # Distance component
  
  update_steps = 25000
  style = harmonic
}
```

**Backward compatibility**: The old `force_constant` parameter is still supported and will be used for the angle component if `angle_force_constant` is not specified.

## Tips for Using Restraints

1. **Strength**: Start with small force constants and increase as needed. Strong restraints can cause simulation instability.

2. **Equilibration**: When using restraints for production runs, first equilibrate the system with the restraints applied.

3. **Atom Indices**: Remember that atom indices are 0-based in FeNNol.

4. **Analysis**: Use the `colvars` section to track the values of restrained coordinates over time:

```
colvars {
  my_distance {
    type = distance
    atom1 = 50
    atom2 = 150
  }
}
```

5. **Monitoring**: The restraint energy is reported separately in the output as `Erestraint`.

## Complete Examples

For complete examples, see:
- `examples/md/aspirin/input_with_restraints.fnl` - Simple molecule with basic restraints
- `examples/md/protein_restraints_example.fnl` - More complex restraints applied to a protein
- `examples/md/backside_attack_example.fnl` - Basic SN2 reaction simulation with angle-only restraints
- `examples/md/backside_attack_composite.fnl` - Advanced SN2 simulation with composite (angle + distance) restraints
- `examples/md/backside_attack_time_varying.fnl` - Time-varying force constants for gradual reaction initiation
- `examples/md/BACKSIDE_ATTACK_SIMULATION_GUIDE.md` - Detailed guide for backside attack simulations