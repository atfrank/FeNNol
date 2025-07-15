# Transition State Search in FeNNol

FeNNol now supports transition state (TS) optimization for finding first-order saddle points on potential energy surfaces. This is useful for studying chemical reactions, conformational changes, and other activated processes.

## Overview

Transition states are saddle points on the potential energy surface - they are minima in all directions except one (the reaction coordinate) where they are maxima. Finding these points is crucial for understanding reaction mechanisms and calculating reaction rates.

## Available Methods

### 1. Quasi-Newton Method

The quasi-Newton method uses Hessian information (either exact or approximate) to find saddle points. It maximizes along the eigenvector with the most negative eigenvalue while minimizing along all other directions.

**Key parameters:**
- `ts_hessian_update`: Hessian update scheme (`bfgs` or `sr1`)
- `ts_trust_radius`: Trust radius for step size control
- `ts_eigenvalue_tolerance`: Tolerance for identifying negative eigenvalues
- `ts_max_uphill_steps`: Maximum consecutive uphill steps allowed

### 2. Dimer Method

The dimer method uses two images separated by a small distance to estimate the lowest curvature mode without requiring Hessian calculations. It's particularly useful for large systems where Hessian storage is prohibitive.

**Key parameters:**
- `dimer_separation`: Distance between dimer images
- `dimer_rotation_tolerance`: Convergence tolerance for dimer rotation
- `dimer_max_rotations`: Maximum rotation iterations per step
- `dimer_initial_mode`: Optional initial orientation for the dimer

### 3. SN2-Specific Method

A specialized method designed for SN2 (Substitution Nucleophilic bimolecular) reactions. This method leverages the well-defined reaction coordinate RC = d(Nu-C) - d(C-LG) to efficiently find the transition state.

**Key parameters:**
- `sn2_nu_index`: 1-based index of nucleophile atom (required)
- `sn2_c_index`: 1-based index of carbon center (required)
- `sn2_lg_index`: 1-based index of leaving group (required)
- `sn2_target_nu_c_distance`: Target Nu-C distance in Å
- `sn2_constraint_strength`: Geometric constraint strength

**See [SN2_TRANSITION_STATES.md](SN2_TRANSITION_STATES.md) for detailed SN2-specific documentation.**

## Usage

### Basic Example

```fnl
# Enable transition state search
transition_state    = True
ts_only            = True           # Exit after TS optimization
ts_method          = quasi_newton   # or "dimer"

# Provide initial guess coordinates
coordinates        = initial_ts_guess.xyz
model_type         = ani2x

# Convergence criteria
min_force_tolerance = 1e-4
min_max_iterations  = 500
```

### Input File Parameters

#### General TS Parameters

- `transition_state`: Enable TS optimization (default: `False`)
- `ts_only`: Only perform TS search, no MD afterwards (default: `False`)
- `ts_method`: Method to use (`quasi_newton` or `dimer`)

#### Quasi-Newton Parameters

- `ts_hessian_update`: `bfgs` (default) or `sr1`
- `ts_max_uphill_steps`: Maximum uphill steps (default: 5)
- `ts_eigenvalue_tolerance`: Eigenvalue tolerance (default: 1e-4)
- `ts_trust_radius`: Trust radius in Å (default: 0.3)
- `ts_initial_hessian_scale`: Initial Hessian diagonal scaling (default: -0.1)

#### Dimer Method Parameters

- `dimer_separation`: Dimer separation in Å (default: 0.01)
- `dimer_rotation_tolerance`: Rotation tolerance (default: 0.1)
- `dimer_max_rotations`: Max rotations per step (default: 10)
- `dimer_rotation_step`: Rotation step size (default: 0.1)
- `dimer_initial_mode`: Initial dimer vector (optional)

## Initial Guess

A good initial guess is crucial for TS optimization. Common approaches include:

1. **Linear interpolation**: Interpolate between reactant and product
2. **Nudged Elastic Band**: Use the highest energy image from a NEB calculation
3. **Chemical intuition**: Manually construct a reasonable TS geometry
4. **Constrained optimization**: Optimize with reaction coordinate fixed

## Output

The TS optimization produces:
- Trajectory file with optimization steps (`.ts.xyz`, `.ts.extxyz`, or `.ts.arc`)
- Final optimized structure
- Energy and force information at each step
- Number of negative eigenvalues (should be 1 for a true TS)
- Lowest eigenvalue value (should be negative)

## Verification

To verify a transition state:
1. Check that there is exactly one negative eigenvalue
2. Verify the negative mode corresponds to the expected reaction coordinate
3. Perform IRC (Intrinsic Reaction Coordinate) calculations
4. Check that optimization from slightly displaced structures returns to the TS

## Example Workflow

```bash
# 1. Prepare initial TS guess
# Create initial_ts_guess.xyz with approximate TS geometry

# 2. Create input file (ts_search.fnl)
cat > ts_search.fnl << EOF
device              = cpu
double_precision    = True
coordinates         = initial_ts_guess.xyz
model_type          = ani2x

transition_state    = True
ts_only            = True
ts_method          = quasi_newton
min_force_tolerance = 1e-4
min_max_iterations  = 500
EOF

# 3. Run TS optimization
fennol_md ts_search.fnl

# 4. Check results
# Look for converged TS with one negative eigenvalue
```

## Tips for Success

1. **Start close**: The initial guess should be reasonably close to the true TS
2. **Check eigenvalues**: Monitor the number of negative eigenvalues during optimization
3. **Adjust trust radius**: Decrease if optimization is unstable, increase if too slow
4. **Try both methods**: If one method fails, try the other
5. **Verify results**: Always verify the TS with frequency calculations or IRC

## Troubleshooting

### Optimization not converging
- Improve initial guess
- Reduce trust radius
- Try different Hessian update scheme
- Switch between quasi-Newton and dimer methods

### Wrong number of negative eigenvalues
- May have found a higher-order saddle point
- Initial guess may be too far from true TS
- Try constraining certain coordinates

### Optimization diverging
- Reduce trust radius
- Check if forces are reasonable
- Verify model is appropriate for system