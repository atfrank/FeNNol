# GB Validation Framework - Executive Summary

**Date**: November 18, 2025
**Purpose**: Quick reference for validation strategy during 10× CUDA optimization
**Full Design**: See [GB_VALIDATION_FRAMEWORK_DESIGN.md](GB_VALIDATION_FRAMEWORK_DESIGN.md)

---

## TL;DR: What Makes This Framework Bulletproof

### 1. Multi-Level Reference Hierarchy

```
Analytical (hand-calculated)
    ↓ validates
NumPy FP64 (slow, obvious)
    ↓ validates
JAX AutoDiff (medium speed)
    ↓ validates
CUDA Baseline (fast, unoptimized)
    ↓ validates
CUDA Optimized (FASTEST, target)
```

**Key Innovation**: Each level validates the one below. If disagreement occurs, we know which implementation to trust.

### 2. Comprehensive Test Coverage

| Test Type | Purpose | System Size | Count | Run Time |
|-----------|---------|-------------|-------|----------|
| **Analytical** | Ground truth | 1-3 atoms | 10 | < 1s |
| **Small** | Detailed validation | 3-20 atoms | 50 | < 30s |
| **Medium** | Scaling validation | 100-1000 atoms | 20 | 5-10 min |
| **Large** | Production validation | 2000-5000 atoms | 5 | 30-60 min |
| **Property-Based** | Automatic edge cases | Random | ∞ | Continuous |
| **Regression** | Known bug prevention | Various | Growing | < 5 min |

**Total**: 85+ hand-crafted + unlimited random cases

### 3. Golden Reference Management

**Problem**: How to validate optimizations without breaking physics?

**Solution**: Version-controlled reference data
- Stored in HDF5 (efficient for numerical arrays)
- Versioned with Git + DVC (Data Version Control)
- Tagged releases (e.g., `golden-v1.0`)
- Automated validation on PR

**Workflow**:
```bash
# Generate golden references once
python scripts/generate_golden_refs.py --version 1.0

# All future optimizations validate against these
pytest tests/test_gb_cuda_optimized.py  # Uses golden-v1.0

# Only update when physics changes (intentionally)
python scripts/update_golden_refs.py --version 1.1 --reason "Fixed OBC scaling"
```

### 4. Property-Based Testing

**Traditional**: Write specific test cases
**Problem**: Miss edge cases, limited coverage

**Property-Based**: Define invariant properties, auto-generate tests
```python
@hypothesis.given(random_system)
def test_energy_permutation_invariance(system):
    """Energy must not change when we swap atoms."""
    assert E(system) == E(permute(system))

@hypothesis.given(random_system, scale_factor)
def test_energy_charge_scaling(system, scale):
    """E(λq) = λ² E(q)"""
    assert E(scale * system.charges) == scale**2 * E(system.charges)

@hypothesis.given(random_system, small_displacement)
def test_force_energy_consistency(system, dr):
    """F·dr = -dE (forces are energy gradient)"""
    assert dot(F, dr) ≈ -(E(x+dr) - E(x))
```

**Result**: Hypothesis explores thousands of random cases, finds edge cases automatically.

### 5. Tolerances Based on Trust Level

```python
# Analytical vs Implementation: TIGHT
ANALYTICAL_TOLERANCE = 1e-10  # Nearly exact

# NumPy Reference vs CUDA: MEDIUM
REFERENCE_TOLERANCE = 1e-6   # FP64 vs FP64, different algorithms

# CUDA Baseline vs Optimized: TIGHT
OPTIMIZATION_TOLERANCE = 1e-8  # Same physics, must match

# OpenMM Cross-Validation: LOOSE
OPENMM_TOLERANCE = 1e-3       # Different code, different choices
```

**Key**: Tighter tolerance for higher trust comparisons.

---

## Quick Start: Validate Your Optimization

### Step 1: Run Baseline Validation

```bash
# Ensure current implementation passes all tests
pytest tests/test_gb_cuda_baseline.py -v

# If any fail, fix before optimizing!
```

### Step 2: Implement Optimization

```cuda
// Your new optimized kernel
__global__ void compute_gb_optimized_kernel(...) {
    // Optimization code here
}
```

### Step 3: Validate Against Baseline

```python
# Add test in tests/test_gb_cuda_optimized.py

def test_my_optimization():
    # Run baseline
    E_base, F_base = compute_gb_baseline(coords, charges, radii, ...)

    # Run optimized
    E_opt, F_opt = compute_gb_optimized(coords, charges, radii, ...)

    # Must match within tight tolerance
    assert np.isclose(E_opt, E_base, atol=1e-8, rtol=1e-7)
    assert np.allclose(F_opt, F_base, atol=1e-6, rtol=1e-5)
```

### Step 4: Run Full Validation Suite

```bash
# All tests
pytest tests/test_gb_*.py -v

# With coverage report
pytest tests/ --cov=fennol.cuda --cov-report=html

# Property-based tests (thorough)
pytest tests/test_gb_properties.py -v --hypothesis-show-statistics
```

### Step 5: Benchmark Performance

```bash
# Run benchmarks
python src/fennol/cuda/tests/benchmark_gb.py

# Compare baseline vs optimized
# Should show ~5-10× speedup with identical results
```

---

## Critical Validation Checkpoints

### Before Committing Code

- [ ] All analytical tests pass (`test_gb_analytical.py`)
- [ ] All reference tests pass (`test_gb_reference.py`)
- [ ] Optimized matches baseline (`test_gb_cuda_optimized.py`)
- [ ] Property-based tests pass (`test_gb_properties.py`)
- [ ] No NaN/Inf in any test case
- [ ] Performance improved (benchmark shows speedup)

### Before Merging PR

- [ ] CI/CD pipeline green (all tests pass)
- [ ] Code review approved
- [ ] Documentation updated
- [ ] Benchmark results added to PR description

### Before Release

- [ ] Full validation suite passes (all test categories)
- [ ] OpenMM cross-validation passes (if available)
- [ ] MD energy conservation test passes
- [ ] Performance regression tests pass
- [ ] Golden references updated (if physics changed)

---

## Common Validation Patterns

### Pattern 1: Analytical Validation (Tiny Systems)

```python
def test_single_atom():
    """Single atom: Born radius = intrinsic radius, no pairwise interaction."""
    coords = [[0, 0, 0]]
    charges = [-0.834]  # Oxygen
    radii = [1.5]

    # Analytical solution
    R_expected = 1.5
    E_expected = -0.5 * (1 - 1/80) * 332.0636 * (-0.834)**2 / 1.5
    F_expected = [[0, 0, 0]]

    # CUDA result
    R_cuda, E_cuda, F_cuda = compute_gb_cuda(coords, charges, radii)

    assert np.isclose(R_cuda[0], R_expected, atol=1e-12)
    assert np.isclose(E_cuda, E_expected, atol=1e-12)
    assert np.allclose(F_cuda, F_expected, atol=1e-12)
```

### Pattern 2: Reference Validation (Small-Medium Systems)

```python
def test_water_box():
    """Water box: validate against NumPy reference."""
    coords, charges, radii = generate_water_box(nx=4, ny=4, nz=4)

    # NumPy reference (slow but correct)
    R_ref, E_ref = compute_gb_numpy_reference(coords, charges, radii)

    # CUDA (fast)
    R_cuda, E_cuda, F_cuda = compute_gb_cuda(coords, charges, radii)

    assert np.allclose(R_cuda, R_ref, atol=1e-6, rtol=1e-5)
    assert np.isclose(E_cuda, E_ref, atol=1e-5, rtol=1e-4)
```

### Pattern 3: Numerical Gradient Validation (Forces)

```python
def test_forces_numerical(coords, charges):
    """Forces must match numerical gradient."""
    E, F_analytical = compute_gb_with_forces(coords, charges)

    # Numerical gradient
    F_numerical = np.zeros_like(coords)
    h = 1e-5
    for i in range(len(coords)):
        for d in range(3):
            coords_plus = coords.copy()
            coords_plus[i, d] += h
            E_plus = compute_gb_energy_only(coords_plus, charges)

            coords_minus = coords.copy()
            coords_minus[i, d] -= h
            E_minus = compute_gb_energy_only(coords_minus, charges)

            F_numerical[i, d] = -(E_plus - E_minus) / (2 * h)

    assert np.allclose(F_analytical, F_numerical, atol=1e-4, rtol=1e-3)
```

### Pattern 4: Optimization Validation (Same Physics)

```python
def test_optimization_preserves_physics():
    """Optimized kernel must give identical results to baseline."""
    coords, charges, radii = load_test_case("water_dimer")

    # Baseline (slow, correct)
    E_base, F_base = compute_gb_baseline(coords, charges, radii)

    # Optimized (fast, should be identical)
    E_opt, F_opt = compute_gb_optimized(coords, charges, radii)

    # Very tight tolerance (same physics!)
    assert np.isclose(E_opt, E_base, atol=1e-8, rtol=1e-7)
    assert np.allclose(F_opt, F_base, atol=1e-6, rtol=1e-5)
```

---

## Innovative Validation Techniques

### 1. Richardson Extrapolation for Optimal Step Size

**Problem**: Finite difference step size too large → truncation error, too small → roundoff error.

**Solution**: Automatically find optimal step size.

```python
def find_optimal_step_size(coords, charges):
    """Find step size that minimizes total error."""
    steps = np.logspace(-8, -3, 20)
    errors = []

    for h in steps:
        # Compute derivative with h
        grad_h = finite_diff(coords, charges, h)

        # Compute derivative with h/2 (more accurate)
        grad_h2 = finite_diff(coords, charges, h/2)

        # Richardson extrapolation error estimate
        error = np.linalg.norm(grad_h - grad_h2)
        errors.append(error)

    optimal_h = steps[np.argmin(errors)]
    return optimal_h
```

### 2. Metamorphic Testing

**Concept**: Even without knowing correct output, we can test relationships.

```python
def test_translation_invariance(coords, charges):
    """Energy must not change when we translate all atoms."""
    E1 = compute_gb_energy(coords, charges)

    # Translate entire system
    coords_translated = coords + np.array([10.0, 20.0, 30.0])
    E2 = compute_gb_energy(coords_translated, charges)

    assert np.isclose(E1, E2, atol=1e-10)

def test_rotation_invariance(coords, charges):
    """Energy must not change when we rotate all atoms."""
    E1 = compute_gb_energy(coords, charges)

    # Random rotation
    R = random_rotation_matrix()
    coords_rotated = coords @ R.T
    E2 = compute_gb_energy(coords_rotated, charges)

    assert np.isclose(E1, E2, atol=1e-10)
```

### 3. Statistical Ensemble Validation (MD Trajectories)

```python
def test_md_ensemble_properties(trajectory):
    """
    Statistical properties of MD trajectory should match theory.

    For NVE ensemble:
    - Energy should be conserved
    - Temperature should fluctuate around target
    - Momentum should be conserved
    """
    energies = [compute_total_energy(frame) for frame in trajectory]

    # Energy conservation
    E_mean = np.mean(energies)
    E_std = np.std(energies)
    E_drift = abs(energies[-1] - energies[0]) / E_mean

    assert E_drift < 0.001, f"Energy drift {E_drift:.2%} too large"
    assert E_std / E_mean < 0.01, f"Energy fluctuation {E_std/E_mean:.2%} too large"

    # Equipartition theorem: <KE> = (3/2) N k_B T
    KE_mean = np.mean([compute_kinetic_energy(frame) for frame in trajectory])
    T_actual = 2 * KE_mean / (3 * len(trajectory[0]) * k_B)

    assert np.isclose(T_actual, T_target, rtol=0.1)
```

### 4. Symbolic Differentiation for Force Validation

**Use JAX for automatic differentiation to validate analytical formulas**:

```python
import jax
import jax.numpy as jnp

def test_forces_vs_jax_autodiff(coords, charges):
    """Compare analytical forces to JAX automatic differentiation."""

    # Define energy function for JAX
    def energy_fn(x):
        return compute_gb_energy_jax(x, charges)

    # Analytical forces (your implementation)
    E, F_analytical = compute_gb_cuda(coords, charges)

    # JAX automatic differentiation
    E_jax = energy_fn(jnp.array(coords))
    F_jax = -jax.grad(energy_fn)(jnp.array(coords))

    # Compare
    assert np.isclose(E, E_jax, atol=1e-6)
    assert np.allclose(F_analytical, F_jax, atol=1e-4, rtol=1e-3)
```

---

## Troubleshooting Guide

### Issue: Test Fails with "Born radii mismatch"

**Diagnosis**:
1. Check HCT integral implementation
2. Verify 0.5*rho scaling is applied
3. Check cutoff handling
4. Look for NaN in psi_sum

**Fix**:
```python
# Add diagnostic prints
print(f"psi_sum: {psi_sum}")
print(f"psi_scaled: {0.5 * rho * psi_sum}")
print(f"tanh_arg: {psi_scaled - b*psi_scaled**2 + c*psi_scaled**3}")
```

### Issue: Energy matches but forces don't

**Diagnosis**: Missing Born radii derivative term!

**Fix**: Ensure forces include:
1. Direct pairwise forces: `∂E/∂r`
2. Born radii forces: `(∂E/∂R_i) * (∂R_i/∂r)`

### Issue: Forces match baseline but not numerical gradient

**Diagnosis**: Both analytical implementations might have same bug.

**Fix**: This is why we use **multi-level validation**. Compare to:
1. JAX autodiff (independent implementation)
2. OpenMM (external code)
3. Analytical derivation (pencil & paper)

### Issue: Random test fails occasionally

**Diagnosis**: Numerical instability or edge case.

**Fix**:
1. Capture the failing random seed
2. Convert to regression test
3. Add safeguards (clamping, epsilon terms)

```python
@given(coords=..., seed=st.integers())
def test_property(coords, seed):
    try:
        result = compute_gb(coords)
        check_property(result)
    except AssertionError as e:
        # Print seed for reproduction
        print(f"Failed with seed {seed}")
        raise
```

---

## Performance vs Correctness Trade-offs

### When to Use Each Validation Level

**Development** (every code change):
- Analytical tests (< 1s)
- Small system tests (< 30s)
- Property-based tests (quick mode)

**Pre-commit** (before git commit):
- All small/medium tests (< 5 min)
- CUDA baseline comparison
- Property-based tests (thorough mode)

**CI/CD** (on PR):
- Full test suite (30-60 min)
- All sizes: analytical → large
- Cross-validation matrix
- Performance benchmarks

**Pre-release** (before version tag):
- Extended MD trajectory tests
- OpenMM cross-validation
- Statistical ensemble validation
- Performance regression suite

---

## Summary: Why This Framework is Bulletproof

1. **Multi-level hierarchy**: If any level fails, we know exactly where the bug is
2. **Comprehensive coverage**: 85+ hand-crafted + ∞ random test cases
3. **Property-based testing**: Finds edge cases automatically
4. **Golden references**: Version-controlled ground truth
5. **Cross-validation**: Multiple independent implementations
6. **Tight tolerances**: Optimization can't silently break physics
7. **Automated CI/CD**: Tests run on every commit
8. **Regression database**: Known bugs stay fixed

**Result**: You can optimize with confidence. If tests pass, physics is preserved.

---

## Next Steps

1. **Week 1-2**: Implement NumPy reference + analytical tests
2. **Week 3-4**: Generate test case library + golden references
3. **Week 5-6**: Validate CUDA baseline + optimize
4. **Week 7-8**: Set up CI/CD + documentation
5. **Week 9-10**: Advanced validation (OpenMM, MD trajectories)

**Start here**: `src/fennol/validation/gb_reference.py`

---

**Full documentation**: [GB_VALIDATION_FRAMEWORK_DESIGN.md](GB_VALIDATION_FRAMEWORK_DESIGN.md)
