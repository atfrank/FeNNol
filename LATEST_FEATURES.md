# FeNNol: New Features and Improvements

## Overview

FeNNol (Force-field-enhanced Neural Networks optimized library) version [e425787] introduces several significant enhancements and bug fixes to improve molecular dynamics simulations. This document outlines the new features, improvements, and bug fixes included in this release.

## 1. Enhanced Restraint System

### 1.1 One-sided Harmonic Restraints

A major addition to FeNNol's restraint system is the implementation of one-sided harmonic restraints, allowing for more flexible control of molecular conformations.

- **Lower-side restraints**: Apply force only when the value is below the target
- **Upper-side restraints**: Apply force only when the value is above the target

**Example Usage:**
```
lower_side_restraint {
  type = distance
  atom1 = 0
  atom2 = 3
  target = 4.0
  force_constant = 50.0
  style = one_sided
  side = lower  # apply force when distance < target
}

upper_side_restraint {
  type = distance
  atom1 = 0
  atom2 = 3
  target = 4.0
  force_constant = 50.0
  style = one_sided
  side = upper  # apply force when distance > target
}
```

### 1.2 Time-varying Restraints

The restraint system now supports time-varying force constants, allowing for gradual introduction or relaxation of restraints during simulation.

- Define force constants as lists of values that change over time
- Control update frequency with the `update_steps` parameter
- Linear interpolation between specified values

**Example Usage:**
```
increasing_force {
  type = distance
  atom1 = 0
  atom2 = 3
  target = 3.5
  force_constant = [0.0, 10.0, 20.0, 50.0, 100.0]  # Values change over time
  update_steps = 100  # Update every 100 steps
  style = harmonic
}
```

### 1.3 Improved Restraint Styles

The restraint system now supports three distinct styles:

- **Harmonic**: Standard harmonic potential (k/2 * (x - x0)²)
- **Flat-bottom**: No energy penalty within a tolerance region, harmonic outside
- **One-sided**: Force applied only when the value is on one side of the target

## 2. Minimization System Improvements

### 2.1 Enhanced Error Handling

The minimization system has been significantly improved with robust error handling:

- Recovery strategies when energy or force calculations fail
- Fallback mechanisms for preprocessing failures
- Proper system state management throughout minimization

### 2.2 Scalar Energy Value Handling

Fixed issues with energy value handling:

- Added `_safe_get_energy_value` method to properly extract scalar energy values
- Consistent energy extraction regardless of return type (array or scalar)
- Proper energy conversion to model units

### 2.3 Simplified Minimization Option

Added a more robust minimization option for complex systems:

- `SimpleSteepestDescentMinimizer` provides more stable optimization
- Direct energy evaluation that bypasses preprocessing for complex systems
- Enabled with the `use_simple_sd` parameter

### 2.4 Advanced Reporting

Improved reporting capabilities during minimization:

- Enhanced trajectory output with energy and force information
- Multiple output format options (xyz, extxyz, arc)
- Detailed convergence statistics and timing information

## 3. Energy Formatting Improvements

### 3.1 Consistent Energy Display

Standardized energy reporting across the codebase:

- Created `format_energy_for_display` function for consistent formatting
- Standardized energy unit conversion across both MD and minimization modules
- Properly handles per-atom energy conversion based on user preferences

### 3.2 Fixed Energy Indexing

Several fixes were made to address energy indexing issues:

- Fixed handling of scalar vs. array energy values
- Updated minimizers to correctly handle energy values
- Prevented index-out-of-bounds errors when accessing energy arrays

### 3.3 Robust Error Handling

Added extensive error handling for energy calculations:

- Implemented try/except blocks around energy calculations
- Added fallback mechanisms for energy extraction errors
- Included type checking to prevent runtime errors

## 4. Collective Variables System

The collective variables system provides a flexible framework for tracking structural properties during simulation:

### 4.1 Supported Collective Variables

- **Distance**: Track the distance between two atoms
- **Angle**: Monitor the angle formed by three atoms
- **Dihedral**: Follow the dihedral angle formed by four atoms

### 4.2 Integration with Simulation

- Values are tracked and recorded throughout simulation
- Output to `.colvars` files for post-processing
- Can be visualized with the provided analysis tools

### 4.3 Analysis Tool

A Python script (`analyze_restraints.py`) is provided for post-processing collective variables data:

- Reads `.colvars` files and extracts time series data
- Generates plots of collective variables vs. time
- Calculates statistics (mean, standard deviation, range, etc.)

## 5. Physics Models Enhancement

### 5.1 NLH Repulsion Model

A major addition to the physics models is the NLH (Nordlund-Lehtola-Hobler) repulsion model:

- Based on research by K. Nordlund, S. Lehtola, G. Hobler: "Repulsive interatomic potentials calculated at three levels of theory"
- Published in Physical Review A 111, 032818 (https://doi.org/10.1103/PhysRevA.111.032818)
- Uses pair-specific coefficients for more accurate repulsion energies:
  - Each atom pair has its own set of coefficients (a1, b1, a2, b2, a3, b3)
  - Covers elements up to Z=92 (uranium)
  - Includes error estimates for each coefficient set
- Functional form similar to ZBL but with element-pair-specific parameters
- More accurate for specific element combinations compared to general ZBL approach

### 5.2 Enhanced ZBL Repulsion Model

The existing ZBL (Ziegler-Biersack-Littmark) repulsion model has been improved:

- Added tunable initial values (exposed as class parameters)
- Improved regularization for training
- Better support for alchemical transformations
- Enhanced parameterization methods

### 5.3 Generalized Graph Softcores

Improvements to the softcore potentials used in molecular dynamics:

- Enhanced handling of alchemical transformations 
- Improved distance calculations with softcore parameters
- More flexible control of lambda parameters for alchemical perturbation 

## 6. Example Files and Documentation

Several example files and documentation have been added:

- `examples/md/RESTRAINTS.md`: Documentation of the restraint system
- `examples/md/minimize_example.fnl`: Example of minimization configuration
- `examples/md/one_sided_restraint_test.fnl`: Demonstration of one-sided restraints
- `examples/md/time_varying_restraint_test.fnl`: Example of time-varying restraints

## 7. Bug Fixes

Multiple bug fixes across the codebase:

- Fixed scalar indexing errors in minimizers
- Corrected energy value handling in various components
- Addressed inconsistencies in energy display
- Fixed potential division by zero in collective variables calculation

## Usage Examples

### Running Minimization with L-BFGS

```
minimize            = True
minimize_only       = True
min_method          = lbfgs
min_max_iterations  = 1000
min_force_tolerance = 1e-4
min_energy_tolerance = 1e-6
min_print_freq      = 1
min_max_step        = 0.2
min_history_size    = 10
```

### Using Flat-bottom Restraints for Protein Backbone

```
ca_restraint {
  type = distance
  atom1 = 10  # CA atom
  atom2 = 10  # Same atom - restrains to initial position
  target = 0.0
  force_constant = 5.0
  style = flat_bottom
  tolerance = 1.0  # Allow movement within 1 Å
}
```

### Creating Time-varying Oscillating Restraints

```
oscillating_restraint {
  type = distance
  atom1 = 5
  atom2 = 10
  target = 3.0
  force_constant = [10.0, 50.0, 10.0, 50.0, 10.0]
  update_steps = 200
  style = harmonic
}
```

## Conclusion

These enhancements significantly improve FeNNol's capabilities for molecular simulations, particularly for systems requiring specialized restraints, careful minimization, and detailed energy analysis. The improved error handling and consistent energy reporting make the library more robust for production simulations, while the new NLH repulsion model offers more accurate interatomic potentials for specific element combinations.