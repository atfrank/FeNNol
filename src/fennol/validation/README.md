# GB Validation Reference Implementations

This directory contains **reference implementations** for validating CUDA GB implicit solvent kernels.

**WARNING**: These implementations prioritize **correctness over performance**. They are intentionally slow but obviously correct. Do NOT use in production MD simulations!

## Purpose

1. **Ground Truth**: Provide bulletproof reference values for test cases
2. **Validation**: Cross-check CUDA optimizations preserve physics
3. **Debugging**: Understand what correct implementation should do
4. **Education**: Learn GB/OBC algorithm step-by-step

## Files

```
validation/
├── README.md                    # This file
├── __init__.py
├── gb_reference.py              # NumPy reference (FP64)
├── gb_numerical_forces.py       # Finite difference forces
├── test_generator.py            # Generate test cases
├── golden_reference.py          # Manage golden reference data
├── tolerances.py                # Tolerance specifications
├── cross_validator.py           # Cross-validation tools
└── openmm_validator.py          # OpenMM cross-validation
```

## Quick Start

### Example 1: Compute Born Radii (Reference)

```python
import numpy as np
from fennol.validation.gb_reference import GBReferenceOBC

# Simple 2-atom system
coords = np.array([
    [0.0, 0.0, 0.0],  # Atom 1
    [2.8, 0.0, 0.0],  # Atom 2
])
radii = np.array([1.5, 1.5])        # Intrinsic radii
b_params = np.array([0.8, 0.8])     # OBC b parameters
c_params = np.array([0.0, 0.0])     # OBC c parameters

# Create reference calculator
ref = GBReferenceOBC(cutoff=12.0)

# Compute Born radii
born_radii, psi_sum = ref.compute_born_radii(
    coords, radii, b_params, c_params
)

print(f"Born radii: {born_radii}")
print(f"Descreening sums: {psi_sum}")
```

### Example 2: Compute GB Energy (Reference)

```python
from fennol.validation.gb_reference import GBEnergyReference

# Same system as above
charges = np.array([-0.834, -0.834])  # Oxygen charges

# Create energy calculator
ref_energy = GBEnergyReference(dielectric=80.0)

# Compute energy
energy = ref_energy.compute_energy(coords, charges, born_radii)

print(f"GB Energy: {energy:.6f} kcal/mol")
```

### Example 3: Compute Numerical Forces

```python
from fennol.validation.gb_numerical_forces import GBForcesNumerical

# Create numerical force calculator
ref_forces = GBForcesNumerical(
    radii=radii,
    b_params=b_params,
    c_params=c_params,
    dielectric=80.0,
    cutoff=12.0
)

# Compute forces via finite differences
# WARNING: This is SLOW (N² energy evaluations)
forces = ref_forces.compute_forces_numerical(
    coords, charges, step=1e-5
)

print(f"Forces:\n{forces}")
```

### Example 4: Validate CUDA vs Reference

```python
from fennol import cuda as fennol_cuda
from fennol.validation.cross_validator import CUDAValidator

# Create validator
validator = CUDAValidator()

# Load test case
coords, charges, atomic_numbers = load_test_case("water_dimer")

# Validate CUDA implementation
report = validator.validate_cuda_vs_reference(
    coords, charges, atomic_numbers,
    test_name="water_dimer"
)

# Print report
print(report)
# {
#   'born_radii': {'pass': True, 'max_diff': 1.2e-7},
#   'energy': {'pass': True, 'diff': 3.4e-6},
#   'forces': {'pass': True, 'max_diff': 2.1e-5}
# }
```

## Design Principles

### 1. Explicit is Better Than Implicit

**BAD** (vectorized, hard to verify):
```python
psi = np.sum(I_matrix, axis=1)
```

**GOOD** (explicit loop, easy to verify):
```python
psi = np.zeros(N)
for i in range(N):
    for j in range(N):
        if i != j:
            psi[i] += descreening_integral(r[i,j], rho[i], rho[j])
```

### 2. Double Precision Everywhere

```python
# Always use float64
coords = np.array(coords, dtype=np.float64)
charges = np.array(charges, dtype=np.float64)

# Avoid implicit float32
bad = np.array([1.5, 2.0])  # Might be float32!
good = np.array([1.5, 2.0], dtype=np.float64)  # Guaranteed float64
```

### 3. Defensive Programming

```python
def compute_descreening_integral(r, rho_i, rho_j):
    """Compute HCT integral with extensive checks."""

    # Input validation
    assert r >= 0, f"Distance r={r} must be non-negative"
    assert rho_i > 0, f"Radius rho_i={rho_i} must be positive"
    assert rho_j > 0, f"Radius rho_j={rho_j} must be positive"

    # Handle edge cases explicitly
    if r < 1e-6:
        return 0.0  # Self-interaction or overlapping

    # Avoid division by zero
    lower_bound = max(rho_i, abs(r - rho_j))
    l_ij = 1.0 / (lower_bound + 1e-12)  # Add epsilon

    # Avoid log(0)
    ratio = np.log(max(u_ij / l_ij, 1e-12))

    # Check output
    result = ...
    assert not np.isnan(result), "Result is NaN!"
    assert not np.isinf(result), "Result is Inf!"

    return result
```

### 4. Match Literature Exactly

```python
# HCT integral formula from Hawkins, Cramer, Truhlar (1996)
# Also matches OpenMM ReferenceObc.cpp lines 145-147

# From paper: I = l_ij - u_ij + 0.25*r*(u_ij² - l_ij²)
#                 + 0.5*ln(u_ij/l_ij)/r + 0.25*s_j²/r*(l_ij² - u_ij²)
integral = (
    l_ij - u_ij                          # First term
    + 0.25 * r * (u_ij2 - l_ij2)        # Second term
    + 0.5 * (1.0 / r) * ratio           # Third term (log)
    + 0.25 * s_j2 * (1.0 / r) * (l_ij2 - u_ij2)  # Fourth term
)
```

## Common Pitfalls

### Pitfall 1: Missing 0.5*rho Scaling

```python
# WRONG: Direct use of psi_sum
tanh_arg = psi_sum - b * psi_sum**2 + c * psi_sum**3

# CORRECT: Scale psi_sum by 0.5*rho first (OpenMM convention)
psi_scaled = 0.5 * rho * psi_sum
tanh_arg = psi_scaled - b * psi_scaled**2 + c * psi_scaled**3
```

### Pitfall 2: Self-Energy Factor

```python
# WRONG: Only pairwise term
E = gb_factor * sum(q[i]*q[j]/f_GB for i<j)

# CORRECT: Include self-energy
E_self = gb_factor * sum(q[i]**2 / R[i])
E_pair = gb_factor * sum(q[i]*q[j]/f_GB for i<j)
E_total = E_self + E_pair
```

### Pitfall 3: Born Radii Clamping

```python
# WRONG: Born radius can be smaller than intrinsic
R = 1.0 / R_inv

# CORRECT: Clamp to intrinsic radius
R = max(1.0 / R_inv, rho)
```

### Pitfall 4: Force Sign Convention

```python
# WRONG: Positive gradient
forces[i] = dE_dx[i]

# CORRECT: Force = -gradient
forces[i] = -dE_dx[i]
```

## Testing Your Reference Implementation

### Test 1: Single Atom

```python
def test_single_atom():
    """Single atom: Born radius = intrinsic, energy = self-energy only."""
    coords = np.array([[0.0, 0.0, 0.0]])
    charges = np.array([-0.834])
    radii = np.array([1.5])
    b_params = np.array([0.8])
    c_params = np.array([0.0])

    ref = GBReferenceOBC(cutoff=12.0)
    born_radii, psi_sum = ref.compute_born_radii(coords, radii, b_params, c_params)

    # Born radius should equal intrinsic (no descreening)
    assert np.isclose(born_radii[0], 1.5, atol=1e-12)

    # psi_sum should be zero (no other atoms)
    assert np.isclose(psi_sum[0], 0.0, atol=1e-12)

    # Energy should be self-energy only
    ref_energy = GBEnergyReference(dielectric=80.0)
    energy = ref_energy.compute_energy(coords, charges, born_radii)

    gb_factor = -0.5 * (1 - 1/80.0) * 332.0636
    E_expected = gb_factor * charges[0]**2 / born_radii[0]

    assert np.isclose(energy, E_expected, atol=1e-12)
```

### Test 2: Two Atoms Far Apart

```python
def test_two_atoms_beyond_cutoff():
    """Two atoms beyond cutoff: no interaction."""
    coords = np.array([
        [0.0, 0.0, 0.0],
        [15.0, 0.0, 0.0]  # Beyond 12 Å cutoff
    ])
    charges = np.array([-0.834, -0.834])
    radii = np.array([1.5, 1.5])
    b_params = np.array([0.8, 0.8])
    c_params = np.array([0.0, 0.0])

    ref = GBReferenceOBC(cutoff=12.0)
    born_radii, psi_sum = ref.compute_born_radii(coords, radii, b_params, c_params)

    # Born radii should be intrinsic (no descreening)
    assert np.allclose(born_radii, radii, atol=1e-12)

    # Energy should be 2 × self-energy (no pairwise)
    ref_energy = GBEnergyReference(dielectric=80.0)
    energy = ref_energy.compute_energy(coords, charges, born_radii)

    gb_factor = -0.5 * (1 - 1/80.0) * 332.0636
    E_expected = gb_factor * np.sum(charges**2 / born_radii)

    assert np.isclose(energy, E_expected, atol=1e-10)
```

### Test 3: Energy Permutation Invariance

```python
def test_permutation_invariance():
    """Energy must not change when we swap atoms."""
    np.random.seed(42)
    coords = np.random.uniform(-5, 5, (10, 3))
    charges = np.random.uniform(-1, 1, 10)
    radii = np.random.uniform(1.0, 2.0, 10)
    b_params = np.full(10, 0.8)
    c_params = np.full(10, 0.0)

    # Compute energy
    ref_born = GBReferenceOBC(cutoff=12.0)
    ref_energy = GBEnergyReference(dielectric=80.0)

    born_radii1, _ = ref_born.compute_born_radii(coords, radii, b_params, c_params)
    E1 = ref_energy.compute_energy(coords, charges, born_radii1)

    # Permute atoms
    perm = np.random.permutation(10)
    coords_perm = coords[perm]
    charges_perm = charges[perm]
    radii_perm = radii[perm]
    b_perm = b_params[perm]
    c_perm = c_params[perm]

    born_radii2, _ = ref_born.compute_born_radii(coords_perm, radii_perm, b_perm, c_perm)
    E2 = ref_energy.compute_energy(coords_perm, charges_perm, born_radii2)

    # Must be identical
    assert np.isclose(E1, E2, atol=1e-10)
```

## Performance Notes

**Reference implementations are SLOW by design!**

| System Size | Born Radii | Energy | Numerical Forces |
|-------------|------------|--------|------------------|
| 10 atoms    | ~1 ms      | ~1 ms  | ~100 ms          |
| 100 atoms   | ~100 ms    | ~10 ms | ~10 s            |
| 1000 atoms  | ~10 s      | ~1 s   | ~1000 s          |

**Do NOT use for**:
- Production MD simulations
- Benchmarking
- Real-time applications

**DO use for**:
- Generating test case reference values (once)
- Validating small systems (N < 100)
- Debugging CUDA implementations
- Understanding GB algorithm

## When Reference Disagrees with CUDA

### Debug Workflow

1. **Test on smallest possible system**
   - Single atom
   - Two atoms
   - Three atoms

2. **Test individual components**
   - Born radii only
   - Energy only
   - Forces only

3. **Add diagnostic prints**
   ```python
   print(f"psi_sum: {psi_sum}")
   print(f"psi_scaled: {psi_scaled}")
   print(f"tanh_arg: {tanh_arg}")
   print(f"born_radii: {born_radii}")
   ```

4. **Compare intermediate values**
   - Descreening integrals
   - f_GB values
   - Born radii derivatives

5. **Check for numerical issues**
   - NaN/Inf in any variable
   - Division by zero
   - Log of negative/zero

6. **Consult literature**
   - Still et al. (1990) - Original GB
   - Onufriev et al. (2004) - OBC variant
   - Hawkins et al. (1996) - HCT integrals
   - OpenMM source code - Reference implementation

## Resources

### Literature

1. **Still et al. (1990)**: "Semianalytical Treatment of Solvation for Molecular Mechanics and Dynamics"
   - Original GB formulation
   - f_GB function definition

2. **Onufriev et al. (2004)**: "Exploring protein native states and large-scale conformational changes with a modified generalized Born model"
   - OBC variant
   - α, β, γ parameters

3. **Hawkins et al. (1996)**: "Pairwise solute descreening of solute charges from a dielectric medium"
   - HCT integral formula
   - Implementation details

### Code References

1. **OpenMM**: `platforms/reference/src/ReferenceObc.cpp`
   - Gold standard reference
   - Match this exactly!

2. **Amber**: `sander/egb.F90`
   - Original OBC implementation
   - Historical reference

3. **GROMACS**: `src/gromacs/gmxlib/nonbonded/nb_kernel_c/nb_generic_cg.c`
   - Alternative implementation
   - Performance optimizations

## Getting Help

**Questions about validation framework**:
- See [GB_VALIDATION_FRAMEWORK_DESIGN.md](../../docs/GB_VALIDATION_FRAMEWORK_DESIGN.md)
- See [GB_VALIDATION_EXECUTIVE_SUMMARY.md](../../docs/GB_VALIDATION_EXECUTIVE_SUMMARY.md)

**Questions about GB physics**:
- See [DEVELOPER_GUIDE_GB_IMPLICIT_SOLVENT.md](../../docs/DEVELOPER_GUIDE_GB_IMPLICIT_SOLVENT.md)
- See [GB_BORN_RADII_FORCES_IMPLEMENTATION.md](../../docs/GB_BORN_RADII_FORCES_IMPLEMENTATION.md)

**Questions about CUDA implementation**:
- See [CUDA_10X_OPTIMIZATION_STRATEGY.md](../../CUDA_10X_OPTIMIZATION_STRATEGY.md)
- See [CUDA_IMPLEMENTATION_SUMMARY.md](../../CUDA_IMPLEMENTATION_SUMMARY.md)

**Bug reports / feature requests**:
- Open GitHub issue with tag `validation`
- Include minimal reproducing example
- Attach test case JSON file

---

**Remember**: These implementations are intentionally simple and slow. Their job is to be obviously correct, not fast!
