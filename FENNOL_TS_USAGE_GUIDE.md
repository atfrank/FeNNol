# FeNNol Transition State Search Usage Guide

## Overview
The `fennol_ts` command-line tool provides transition state optimization using neural network potentials. This guide covers proper usage, best practices, and troubleshooting.

## Installation & Setup

### Prerequisites
- JAX with CPU or GPU support
- FeNNol package installed
- MACE model file (e.g., `mace_mp_large.fnx`)

### Environment Setup
```bash
# For CPU-only execution
export JAX_PLATFORMS=cpu

# For GPU execution (if available)
export JAX_PLATFORMS=cuda
```

## Basic Usage

### Command Syntax
```bash
fennol_ts [OPTIONS] input_file.fnl
```

### Input File Format
Create a `.fnl` parameter file with the following structure:

```bash
# Device and precision settings
device = cpu
enable_x64 = True
matmul_prec = highest

# Model file path
model_file = /path/to/your/model.fnx

# System input
xyz_input {
    file = system.xyz
    indexed = no
    has_comment_line = yes
}

# Transition state settings
transition_state = True
ts_only = True
ts_method = quasi_newton
ts_hessian_update = bfgs
ts_trust_radius = 0.2
ts_eigenvalue_tolerance = 1e-4
ts_max_uphill_steps = 2
ts_initial_hessian_scale = 0.05

# Optimization parameters
min_max_iterations = 20
min_force_tolerance = 1e-3
min_print_freq = 1
min_max_step = 0.3

# Output settings
output_prefix = my_system
traj_format = xyz
```

### Example Commands

#### Basic TS Search
```bash
# Run basic transition state search
fennol_ts system.fnl

# Run with verbose output
fennol_ts system.fnl --verbose

# Override method and iterations
fennol_ts system.fnl --method quasi_newton --max-iterations 50
```

#### Trajectory Control
```bash
# Disable multi-model trajectory
fennol_ts system.fnl --no-multimodel

# Disable PDB trajectory
fennol_ts system.fnl --no-pdb

# Both disabled
fennol_ts system.fnl --no-multimodel --no-pdb
```

## Available Methods

### 1. Quasi-Newton Method (`quasi_newton`)
**Best for**: General transition state searches, well-behaved systems
```bash
ts_method = quasi_newton
ts_hessian_update = bfgs           # or sr1
ts_trust_radius = 0.2              # Å
ts_eigenvalue_tolerance = 1e-4
ts_max_uphill_steps = 2
ts_initial_hessian_scale = 0.05
```

### 2. Dimer Method (`dimer`)
**Best for**: Systems where Hessian updates are problematic
```bash
ts_method = dimer
dimer_separation = 0.01            # Å
dimer_rotation_tolerance = 0.1
dimer_max_rotations = 10
dimer_rotation_step = 0.1
```

### 3. SN2 Method (`sn2`)
**Best for**: SN2 reactions with known nucleophile, carbon, and leaving group
```bash
ts_method = sn2
sn2_nu_index = 1                   # Nucleophile atom index (1-based)
sn2_c_index = 2                    # Carbon center atom index
sn2_lg_index = 3                   # Leaving group atom index
sn2_constraint_strength = 0.1
sn2_target_nu_c_distance = 2.0     # Å
sn2_target_c_lg_distance = 2.0     # Å
```

## Output Files

### Standard Output Files
- `system.ts.xyz` - Final optimized transition state structure
- `system.multimodel.xyz` - Multi-model trajectory (all frames)
- `system.traj.pdb` - PDB trajectory with MODEL/ENDMDL sections
- `system.best_ts_guess.xyz` - Best TS structure found during optimization
- `system.best_ts_guess.pdb` - Best TS structure in PDB format

### Trajectory Analysis
The multi-model files contain:
- Energy progression throughout optimization
- Force magnitudes (max and RMS)
- Structural changes at each step
- Metadata in comments/REMARK lines

## Parameter Tuning Guide

### For Converged Results
```bash
# Conservative settings for stability
ts_trust_radius = 0.1              # Smaller steps
ts_max_uphill_steps = 1            # Fewer uphill steps
min_force_tolerance = 5e-4         # Looser convergence
min_max_iterations = 50            # More iterations
```

### For Faster Results
```bash
# Aggressive settings for speed
ts_trust_radius = 0.3              # Larger steps
ts_max_uphill_steps = 3            # More uphill steps
min_force_tolerance = 1e-3         # Standard convergence
min_max_iterations = 30            # Fewer iterations
```

### For Difficult Systems
```bash
# Robust settings for problematic cases
ts_method = dimer                  # More robust method
ts_trust_radius = 0.05             # Very small steps
ts_initial_hessian_scale = 0.02    # Smaller initial Hessian
min_max_iterations = 100           # Many iterations
```

## Best Practices

### 1. Initial Structure Preparation
- Start with a reasonable guess structure
- Pre-optimize reactants and products
- Ensure proper atomic connectivity
- Check for reasonable bond lengths/angles

### 2. Method Selection
- **Quasi-Newton**: General purpose, fastest convergence
- **Dimer**: More robust, better for difficult cases
- **SN2**: Specialized for SN2 reactions with constraints

### 3. Convergence Monitoring
- Monitor energy progression (should be controlled)
- Check force magnitudes (should decrease overall)
- Verify single negative eigenvalue (N_Neg_Eval = 1)
- Examine structural changes (should be reasonable)

### 4. Result Validation
- Analyze final structure chemically
- Check bond lengths and angles
- Verify reaction coordinate
- Compare with experimental/literature data

## Troubleshooting

### Common Issues

#### 1. **Optimization Divergence**
```
ERROR: Energy increases rapidly, forces explode
```
**Solution**: Reduce trust radius, use more conservative settings
```bash
ts_trust_radius = 0.1
ts_max_uphill_steps = 1
ts_initial_hessian_scale = 0.02
```

#### 2. **Too Many Negative Eigenvalues**
```
WARNING: Too many negative eigenvalues (>2), resetting Hessian
```
**Solution**: This is handled automatically, but indicates problematic Hessian
```bash
ts_method = dimer  # Switch to dimer method
```

#### 3. **Slow Convergence**
```
WARNING: High final max force after many iterations
```
**Solution**: Loosen convergence criteria or increase iterations
```bash
min_force_tolerance = 5e-4
min_max_iterations = 100
```

#### 4. **Skin Update Errors**
```
Warning: Error in skin update: 'edge_src_skin'
```
**Solution**: These are handled gracefully, no action needed

### Expected Behavior

#### Normal Optimization
- Energy increases gradually and controlled
- Forces increase initially, then may stabilize
- Exactly 1 negative eigenvalue maintained
- Structural changes are smooth and reasonable

#### Acceptable Results
- **Energy range**: -100 to +50 kcal/mol/atom (depends on system)
- **Final forces**: < 5.0 (ideally < 1.0 for convergence)
- **Structural changes**: 0.1-0.5 Å RMSD total displacement
- **Eigenvalues**: Exactly 1 negative eigenvalue

## Analysis Tools

### Structural Analysis
```bash
# Run analysis on trajectory
python analyze_structural_changes.py

# Check chemical correctness
python analyze_ts_results.py
```

### Manual Validation
```bash
# Check output files
ls -la system.*

# Examine trajectory
cat system.multimodel.xyz

# View in molecular viewer
# Open system.ts.xyz in VMD, PyMOL, etc.
```

## Performance Tips

### CPU Optimization
```bash
# Set CPU-only execution
export JAX_PLATFORMS=cpu

# Use appropriate precision
enable_x64 = True              # Higher precision
matmul_prec = highest          # Best accuracy
```

### GPU Optimization (if available)
```bash
# Use GPU acceleration
export JAX_PLATFORMS=cuda
device = gpu
```

### Memory Management
- Use smaller systems for testing
- Limit trajectory output for large systems
- Monitor memory usage during optimization

## Integration with Workflows

### Reaction Path Studies
1. Optimize reactants/products separately
2. Use transition state as starting point
3. Perform IRC calculations (if available)
4. Validate with thermodynamic analysis

### High-Throughput Screening
```bash
# Process multiple systems
for system in system1 system2 system3; do
    fennol_ts ${system}.fnl --max-iterations 30
done
```

## Citation & Support

When using FeNNol transition state search in publications, please cite:
- FeNNol package
- MACE model (if used)
- Relevant method papers

For support:
- Check this usage guide
- Review error messages and warnings
- Examine trajectory files for debugging
- Contact development team if needed

---

**Note**: This tool finds transition state guesses using neural network potentials. Always validate results with appropriate quantum chemistry methods for publication-quality work.