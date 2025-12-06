# GB Implicit Solvent Validation Framework Design

**Date**: November 18, 2025
**Purpose**: Comprehensive validation system for CUDA GB kernel optimizations
**Target**: Ensure 10× performance optimization preserves physics accuracy

---

## Table of Contents

1. [Reference Implementation Strategy](#1-reference-implementation-strategy)
2. [Test Case Generation System](#2-test-case-generation-system)
3. [Golden Reference Data Management](#3-golden-reference-data-management)
4. [Cross-Validation Strategy](#4-cross-validation-strategy)
5. [Automated Testing Framework](#5-automated-testing-framework)
6. [Implementation Roadmap](#6-implementation-roadmap)

---

## 1. Reference Implementation Strategy

### 1.1 Design Philosophy

**Goal**: Create bulletproof reference implementations that are:
- **Obviously correct** (simple, readable code)
- **Numerically stable** (FP64, careful algorithm design)
- **Well-tested** (against analytical solutions and OpenMM)
- **Slow but reliable** (not optimized, prioritize correctness)

### 1.2 Reference Implementation Hierarchy

```
Level 1: Analytical Solutions (for tiny systems)
  └─> Level 2: NumPy Reference (Python, FP64)
      └─> Level 3: JAX Reference (for autodiff validation)
          └─> Level 4: CUDA Baseline (unoptimized)
              └─> Level 5: CUDA Optimized (target)
```

**Validation chain**: Each level validates against the level above it.

### 1.3 Component-Specific References

#### 1.3.1 Born Radii Calculation (OBC Model)

**Reference Implementation**: Python NumPy (FP64)

```python
# File: src/fennol/validation/gb_reference.py

import numpy as np
from typing import Tuple

class GBReferenceOBC:
    """
    Reference implementation of OBC Born radii calculation.

    This is intentionally SLOW but CORRECT. Uses:
    - Double precision (FP64) everywhere
    - Explicit loops (no vectorization)
    - Direct implementation of HCT integral formula
    - Extensive validation checks

    DO NOT optimize this code! It's the reference.
    """

    def __init__(self, cutoff: float = 12.0):
        self.cutoff = cutoff

    def compute_descreening_integral_HCT(
        self,
        r: float,
        rho_i: float,
        rho_j: float
    ) -> float:
        """
        Compute HCT descreening integral I(r, ρ_i, ρ_j).

        Reference: Hawkins, Cramer, Truhlar (1996)
        Matches: OpenMM ReferenceObc.cpp

        Args:
            r: Distance between atoms i and j (Angstroms)
            rho_i: Intrinsic radius of atom i
            rho_j: Intrinsic radius of atom j (= s_j in literature)

        Returns:
            Descreening integral value
        """
        # Handle edge cases
        if r < 1e-6:  # Self-interaction or overlapping
            return 0.0

        s_j = rho_j  # OpenMM notation

        # Geometric bounds
        upper_limit = rho_i + s_j
        lower_limit = abs(rho_i - s_j)

        # Region 1: No overlap (r >= upper_limit)
        if r >= upper_limit:
            return 0.0

        # Region 2: Complete or partial overlap (r < upper_limit)
        # Clamp r to lower_limit for complete overlap region
        r_calc = max(r, lower_limit)

        # Integration bounds (HCT formula)
        # l_ij = 1/max(ρ_i, |r - s_j|)
        abs_diff = abs(r_calc - s_j)
        lower_bound = max(rho_i, abs_diff)
        l_ij = 1.0 / lower_bound

        # u_ij = 1/(r + s_j)
        u_ij = 1.0 / (r_calc + s_j)

        # Precompute powers
        l_ij2 = l_ij * l_ij
        u_ij2 = u_ij * u_ij
        s_j2 = s_j * s_j
        r_inv = 1.0 / r_calc

        # Avoid log(0)
        ratio = np.log(max(u_ij / l_ij, 1e-12))

        # HCT integral formula (OpenMM ReferenceObc.cpp line 145-147):
        # I = l_ij - u_ij + 0.25*r*(u_ij² - l_ij²)
        #     + 0.5*ln(u_ij/l_ij)/r + 0.25*s_j²/r*(l_ij² - u_ij²)
        integral = (
            l_ij - u_ij
            + 0.25 * r_calc * (u_ij2 - l_ij2)
            + 0.5 * r_inv * ratio
            + 0.25 * s_j2 * r_inv * (l_ij2 - u_ij2)
        )

        return integral

    def compute_born_radii(
        self,
        coords: np.ndarray,      # [N, 3]
        radii: np.ndarray,       # [N]
        b_params: np.ndarray,    # [N]
        c_params: np.ndarray     # [N]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Born radii using OBC model.

        Returns:
            born_radii: Effective Born radii [N]
            psi_sum: Descreening sums [N] (for force calculation)
        """
        N = len(coords)
        psi_sum = np.zeros(N, dtype=np.float64)

        # Compute pairwise descreening (explicit O(N²) loop)
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                # Compute distance
                dr = coords[i] - coords[j]
                r = np.linalg.norm(dr)

                # Apply cutoff
                if r > self.cutoff:
                    continue

                # Accumulate descreening integral
                I_ij = self.compute_descreening_integral_HCT(
                    r, radii[i], radii[j]
                )
                psi_sum[i] += I_ij

        # Apply OBC formula to get Born radii
        born_radii = np.zeros(N, dtype=np.float64)

        for i in range(N):
            rho_i = radii[i]
            psi_i = psi_sum[i]

            # CRITICAL: Scale psi by 0.5*rho_i (OpenMM convention)
            psi_scaled = 0.5 * rho_i * psi_i

            # OBC tanh polynomial
            psi_2 = psi_scaled * psi_scaled
            psi_3 = psi_2 * psi_scaled
            tanh_arg = psi_scaled - b_params[i] * psi_2 + c_params[i] * psi_3
            tanh_val = np.tanh(tanh_arg)

            # Born radius formula: 1/R = 1/rho - tanh(...)/rho
            R_inv = 1.0 / rho_i - tanh_val / rho_i
            R = 1.0 / R_inv

            # Clamp to intrinsic radius
            born_radii[i] = max(R, rho_i)

        return born_radii, psi_sum
```

**Usage**:
```python
# For unit tests only - this is SLOW!
ref = GBReferenceOBC(cutoff=12.0)
born_radii_ref, psi_ref = ref.compute_born_radii(coords, radii, b, c)
```

#### 1.3.2 GB Energy Calculation

**Reference Implementation**: Python NumPy (FP64)

```python
class GBEnergyReference:
    """Reference GB electrostatic energy calculation."""

    COULOMB_CONST = 332.0636  # kcal·Å·mol⁻¹·e⁻²

    def __init__(self, dielectric: float = 80.0):
        self.dielectric = dielectric
        # GB factor: -0.5 * (1 - 1/ε) * COULOMB
        self.gb_factor = -0.5 * (1.0 - 1.0 / dielectric) * self.COULOMB_CONST

    def compute_f_GB(
        self,
        r: float,
        R_i: float,
        R_j: float
    ) -> float:
        """
        Compute GB screening function.

        f_GB(r, R_i, R_j) = sqrt(r² + R_i*R_j * exp(-r²/(4*R_i*R_j)))

        Reference: Still et al. (1990)
        """
        R_product = R_i * R_j
        r_sq = r * r
        exp_arg = -r_sq / (4.0 * R_product)
        exp_term = np.exp(exp_arg)
        f_gb = np.sqrt(r_sq + R_product * exp_term)
        return f_gb

    def compute_energy(
        self,
        coords: np.ndarray,       # [N, 3]
        charges: np.ndarray,      # [N]
        born_radii: np.ndarray    # [N]
    ) -> float:
        """
        Compute GB electrostatic solvation energy.

        E_GB = gb_factor * (E_self + E_pair)

        where:
        - E_self = Σ_i q_i²/R_i (self-solvation)
        - E_pair = Σ_i<j q_i*q_j/f_GB(r_ij, R_i, R_j) (pairwise)
        """
        N = len(coords)

        # Self-energy term
        E_self = np.sum(charges**2 / born_radii)

        # Pairwise energy term (explicit sum over i < j)
        E_pair = 0.0
        for i in range(N):
            for j in range(i + 1, N):  # Only upper triangle
                dr = coords[i] - coords[j]
                r = np.linalg.norm(dr)

                f_gb = self.compute_f_GB(r, born_radii[i], born_radii[j])
                E_pair += charges[i] * charges[j] / f_gb

        # Total GB energy
        E_total = self.gb_factor * (E_self + E_pair)

        return E_total
```

#### 1.3.3 GB Forces - Numerical Differentiation

**Reference Implementation**: Finite Difference (FP64)

```python
class GBForcesNumerical:
    """
    Numerical gradient computation for GB forces.

    This is the ULTIMATE reference - if analytical forces disagree
    with this, the analytical forces are wrong!

    Uses central finite differences with careful step size selection.
    """

    def __init__(
        self,
        radii: np.ndarray,
        b_params: np.ndarray,
        c_params: np.ndarray,
        dielectric: float = 80.0,
        cutoff: float = 12.0
    ):
        self.born_calculator = GBReferenceOBC(cutoff)
        self.energy_calculator = GBEnergyReference(dielectric)
        self.radii = radii
        self.b_params = b_params
        self.c_params = c_params

    def compute_energy_at_coords(
        self,
        coords: np.ndarray,
        charges: np.ndarray
    ) -> float:
        """Compute total GB energy at given coordinates."""
        # Recompute Born radii at these coordinates
        born_radii, _ = self.born_calculator.compute_born_radii(
            coords, self.radii, self.b_params, self.c_params
        )

        # Compute energy
        energy = self.energy_calculator.compute_energy(
            coords, charges, born_radii
        )

        return energy

    def compute_forces_numerical(
        self,
        coords: np.ndarray,
        charges: np.ndarray,
        step: float = 1e-5
    ) -> np.ndarray:
        """
        Compute forces using central finite differences.

        F_i = -∂E/∂x_i ≈ -(E(x+h) - E(x-h))/(2h)

        Args:
            coords: Coordinates [N, 3]
            charges: Charges [N]
            step: Finite difference step size (Angstroms)

        Returns:
            forces: Forces [N, 3] in kcal/(mol·Å)
        """
        N = len(coords)
        forces = np.zeros((N, 3), dtype=np.float64)

        for i in range(N):
            for d in range(3):
                # Forward step
                coords_plus = coords.copy()
                coords_plus[i, d] += step
                E_plus = self.compute_energy_at_coords(coords_plus, charges)

                # Backward step
                coords_minus = coords.copy()
                coords_minus[i, d] -= step
                E_minus = self.compute_energy_at_coords(coords_minus, charges)

                # Central difference
                forces[i, d] = -(E_plus - E_minus) / (2.0 * step)

        return forces

    def estimate_optimal_step_size(
        self,
        coords: np.ndarray,
        charges: np.ndarray
    ) -> float:
        """
        Estimate optimal finite difference step size.

        Uses Richardson extrapolation to find step size that
        minimizes truncation + roundoff error.

        Returns:
            Optimal step size in Angstroms
        """
        # Test different step sizes on first atom, x-component
        steps = np.logspace(-8, -3, 20)
        errors = []

        for h in steps:
            # Compute derivative with step h
            coords_plus = coords.copy()
            coords_plus[0, 0] += h
            E_plus = self.compute_energy_at_coords(coords_plus, charges)

            coords_minus = coords.copy()
            coords_minus[0, 0] -= h
            E_minus = self.compute_energy_at_coords(coords_minus, charges)

            deriv_h = (E_plus - E_minus) / (2 * h)

            # Compute derivative with step h/2 (higher accuracy)
            coords_plus2 = coords.copy()
            coords_plus2[0, 0] += h / 2
            E_plus2 = self.compute_energy_at_coords(coords_plus2, charges)

            coords_minus2 = coords.copy()
            coords_minus2[0, 0] -= h / 2
            E_minus2 = self.compute_energy_at_coords(coords_minus2, charges)

            deriv_h2 = (E_plus2 - E_minus2) / h

            # Richardson extrapolation error estimate
            error = abs(deriv_h - deriv_h2)
            errors.append(error)

        # Find step size with minimum error
        optimal_idx = np.argmin(errors)
        optimal_step = steps[optimal_idx]

        return optimal_step
```

### 1.4 Reference Implementation Selection Matrix

| Component | Language | Precision | Speed | Use Case |
|-----------|----------|-----------|-------|----------|
| **Born Radii** | Python/NumPy | FP64 | Slow | Unit tests (N < 100) |
| **GB Energy** | Python/NumPy | FP64 | Slow | Unit tests (N < 100) |
| **GB Forces** | Finite Diff | FP64 | Very Slow | Validation (N < 20) |
| **Analytical Forces** | JAX AutoDiff | FP64 | Medium | Cross-check (N < 1000) |
| **CUDA Baseline** | CUDA Unopt | FP64 | Fast | Regression (N < 5000) |

### 1.5 When to Use Which Reference

```
Small systems (N < 10):
├─> Analytical solutions (hand-calculated)
├─> NumPy reference
└─> Numerical forces (finite differences)

Medium systems (N = 100-1000):
├─> NumPy reference (Born radii, energy)
├─> JAX autodiff (forces)
└─> Compare CUDA vs JAX

Large systems (N > 1000):
├─> CUDA baseline vs optimized
├─> Energy conservation checks
└─> Statistical validation (MD trajectories)

Protein MD (N = 2000-5000):
└─> OpenMM cross-validation (if available)
```

---

## 2. Test Case Generation System

### 2.1 Test Case Categories

```
test_cases/
├── analytical/          # Hand-solvable systems
│   ├── single_atom.json
│   ├── two_atoms_*.json
│   └── three_atoms_*.json
│
├── small/               # 3-10 atoms
│   ├── water_monomer.json
│   ├── water_dimer_*.json
│   ├── charged_pairs_*.json
│   └── edge_cases/
│       ├── atoms_at_cutoff.json
│       ├── overlapping_atoms.json
│       ├── highly_charged.json
│       └── zero_charge.json
│
├── medium/              # 100-1000 atoms
│   ├── water_boxes/
│   │   ├── water_64.json
│   │   ├── water_216.json
│   │   └── water_512.json
│   ├── small_peptides/
│   │   ├── ala_dipeptide.json
│   │   ├── trp_cage.json
│   │   └── villin_headpiece.json
│   └── mixed_systems/
│       ├── protein_water.json
│       └── ions_water.json
│
├── large/               # 2000-5000 atoms
│   ├── proteins/
│   │   ├── dhfr.json
│   │   ├── lysozyme.json
│   │   └── ubiquitin.json
│   └── stress_tests/
│       ├── random_configs_*.json
│       └── extreme_geometries.json
│
└── regression/          # Known failure cases (for CI)
    ├── issue_42_nan_forces.json
    └── issue_67_energy_drift.json
```

### 2.2 Test Case Data Format

**JSON Schema**:
```json
{
  "metadata": {
    "name": "water_dimer_equilibrium",
    "description": "Two water molecules at equilibrium distance",
    "category": "small",
    "date_created": "2025-11-18",
    "created_by": "test_generator_v1.0",
    "tags": ["water", "analytical", "equilibrium"]
  },

  "system": {
    "natoms": 6,
    "coordinates": [
      [0.0, 0.0, 0.0],
      [0.757, 0.586, 0.0],
      ...
    ],
    "charges": [-0.834, 0.417, ...],
    "atomic_numbers": [8, 1, 1, 8, 1, 1],
    "box": null  // or [Lx, Ly, Lz] for periodic
  },

  "parameters": {
    "radii_set": "mbondi",
    "b_params": [0.8, 0.8, ...],
    "c_params": [0.0, 0.0, ...],
    "dielectric": 80.0,
    "cutoff": 12.0,
    "surface_tension": 0.005,
    "probe_radius": 1.4
  },

  "reference_values": {
    "born_radii": [1.824, 1.268, ...],
    "psi_sum": [0.127, 0.089, ...],
    "energy": -12.456,
    "forces": [
      [-0.123, 0.456, 0.0],
      ...
    ],
    "max_force": 1.234,

    "provenance": {
      "method": "numpy_reference_v1.0",
      "precision": "float64",
      "date": "2025-11-18",
      "code_hash": "a1b2c3d4"
    },

    "tolerances": {
      "born_radii_abs": 1e-6,
      "energy_abs": 1e-5,
      "forces_abs": 1e-4,
      "forces_rel": 1e-3
    }
  }
}
```

### 2.3 Test Case Generator

**File**: `src/fennol/validation/test_generator.py`

```python
import numpy as np
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from pathlib import Path

@dataclass
class TestCase:
    """Container for a single GB test case."""
    metadata: Dict
    system: Dict
    parameters: Dict
    reference_values: Optional[Dict] = None

    def to_json(self, filepath: Path):
        """Save test case to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def from_json(cls, filepath: Path):
        """Load test case from JSON file."""
        with open(filepath) as f:
            data = json.load(f)
        return cls(**data)


class TestCaseGenerator:
    """Generate systematic test cases for GB validation."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_single_atom(self) -> TestCase:
        """
        Generate single atom test case.

        For a single atom, Born radius = intrinsic radius.
        Energy = gb_factor * q²/R
        Forces = 0
        """
        coords = np.array([[0.0, 0.0, 0.0]])
        charges = np.array([-0.834])  # Oxygen charge
        atomic_numbers = np.array([8])

        # OBC parameters for oxygen
        radii = np.array([1.5])
        b_params = np.array([0.8])
        c_params = np.array([0.0])

        # Analytical solution
        born_radii = radii.copy()  # No descreening
        energy = -0.5 * (1 - 1/80.0) * 332.0636 * charges[0]**2 / born_radii[0]
        forces = np.zeros((1, 3))

        return TestCase(
            metadata={
                "name": "single_oxygen_atom",
                "description": "Single oxygen atom - analytical solution",
                "category": "analytical",
                "date_created": "2025-11-18"
            },
            system={
                "natoms": 1,
                "coordinates": coords.tolist(),
                "charges": charges.tolist(),
                "atomic_numbers": atomic_numbers.tolist()
            },
            parameters={
                "radii": radii.tolist(),
                "b_params": b_params.tolist(),
                "c_params": c_params.tolist(),
                "dielectric": 80.0,
                "cutoff": 12.0
            },
            reference_values={
                "born_radii": born_radii.tolist(),
                "energy": float(energy),
                "forces": forces.tolist(),
                "provenance": {"method": "analytical"},
                "tolerances": {
                    "born_radii_abs": 1e-12,
                    "energy_abs": 1e-12,
                    "forces_abs": 1e-12
                }
            }
        )

    def generate_two_atoms_far(self, separation: float = 15.0) -> TestCase:
        """
        Two atoms beyond cutoff - no interaction.

        Born radii = intrinsic radii
        Energy = 2 * self-energy (no pairwise term)
        Forces = 0
        """
        coords = np.array([
            [0.0, 0.0, 0.0],
            [separation, 0.0, 0.0]
        ])
        charges = np.array([-0.834, -0.834])
        atomic_numbers = np.array([8, 8])

        radii = np.array([1.5, 1.5])
        b_params = np.array([0.8, 0.8])
        c_params = np.array([0.0, 0.0])

        # Analytical: no interaction beyond cutoff
        born_radii = radii.copy()
        gb_factor = -0.5 * (1 - 1/80.0) * 332.0636
        energy = gb_factor * np.sum(charges**2 / born_radii)
        forces = np.zeros((2, 3))

        return TestCase(
            metadata={
                "name": f"two_atoms_far_sep{separation:.1f}",
                "description": f"Two atoms at {separation} Å (beyond cutoff)",
                "category": "edge_cases"
            },
            system={
                "natoms": 2,
                "coordinates": coords.tolist(),
                "charges": charges.tolist(),
                "atomic_numbers": atomic_numbers.tolist()
            },
            parameters={
                "radii": radii.tolist(),
                "b_params": b_params.tolist(),
                "c_params": c_params.tolist(),
                "dielectric": 80.0,
                "cutoff": 12.0
            },
            reference_values={
                "born_radii": born_radii.tolist(),
                "energy": float(energy),
                "forces": forces.tolist(),
                "provenance": {"method": "analytical"},
                "tolerances": {
                    "born_radii_abs": 1e-12,
                    "energy_abs": 1e-10,
                    "forces_abs": 1e-10
                }
            }
        )

    def generate_water_box(self, nx: int, ny: int, nz: int) -> TestCase:
        """
        Generate water box test case.

        Args:
            nx, ny, nz: Number of water molecules in each dimension
        """
        from fennol.utils.pdb import generate_water_box  # Hypothetical

        # Generate water box geometry
        coords, charges, atomic_numbers = generate_water_box(
            nx, ny, nz, spacing=3.0
        )

        # Generate reference values using NumPy reference
        from fennol.validation.gb_reference import (
            GBReferenceOBC, GBEnergyReference
        )

        ref_born = GBReferenceOBC(cutoff=12.0)
        ref_energy = GBEnergyReference(dielectric=80.0)

        radii = get_radii_for_atoms(atomic_numbers)
        b_params, c_params = get_obc_params(atomic_numbers)

        born_radii, psi_sum = ref_born.compute_born_radii(
            coords, radii, b_params, c_params
        )

        energy = ref_energy.compute_energy(coords, charges, born_radii)

        # Forces via numerical differentiation (expensive!)
        # Skip for large systems, validate analytically instead
        forces = None

        return TestCase(
            metadata={
                "name": f"water_box_{nx}x{ny}x{nz}",
                "description": f"Water box with {nx*ny*nz} molecules",
                "category": "medium"
            },
            system={
                "natoms": len(coords),
                "coordinates": coords.tolist(),
                "charges": charges.tolist(),
                "atomic_numbers": atomic_numbers.tolist()
            },
            parameters={
                "radii": radii.tolist(),
                "b_params": b_params.tolist(),
                "c_params": c_params.tolist(),
                "dielectric": 80.0,
                "cutoff": 12.0
            },
            reference_values={
                "born_radii": born_radii.tolist(),
                "psi_sum": psi_sum.tolist(),
                "energy": float(energy),
                "forces": forces,
                "provenance": {
                    "method": "numpy_reference",
                    "precision": "float64"
                },
                "tolerances": {
                    "born_radii_abs": 1e-6,
                    "energy_abs": 1e-5,
                    "forces_abs": 1e-4
                }
            }
        )

    def generate_all_test_cases(self):
        """Generate complete test suite."""

        # Analytical cases
        print("Generating analytical test cases...")
        test = self.generate_single_atom()
        test.to_json(self.output_dir / "analytical" / "single_atom.json")

        for sep in [15.0, 20.0, 50.0]:
            test = self.generate_two_atoms_far(sep)
            test.to_json(self.output_dir / "edge_cases" / f"far_atoms_{sep:.0f}.json")

        # Small systems
        print("Generating small system test cases...")
        # ... water dimers at different separations
        # ... charged pairs
        # ... edge cases

        # Medium systems
        print("Generating medium system test cases...")
        for nx, ny, nz in [(4,4,4), (6,6,6), (8,8,8)]:
            test = self.generate_water_box(nx, ny, nz)
            natoms = nx * ny * nz * 3
            test.to_json(self.output_dir / "medium" / f"water_{natoms}.json")

        print(f"Test cases saved to {self.output_dir}")
```

### 2.4 Property-Based Test Generation

**Random test case generation with invariant checking**:

```python
import hypothesis
from hypothesis import given, strategies as st
import hypothesis.extra.numpy as npst

class PropertyBasedTestGenerator:
    """Generate random test cases and check invariant properties."""

    @given(
        natoms=st.integers(min_value=2, max_value=20),
        box_size=st.floats(min_value=10.0, max_value=50.0),
        charge_scale=st.floats(min_value=0.1, max_value=2.0)
    )
    def test_energy_symmetry(self, natoms, box_size, charge_scale):
        """
        Property: GB energy must be symmetric under atom permutation.

        If we swap atoms i and j (coordinates, charges, radii),
        energy should remain the same.
        """
        # Generate random configuration
        coords = np.random.uniform(-box_size/2, box_size/2, (natoms, 3))
        charges = np.random.uniform(-charge_scale, charge_scale, natoms)
        radii = np.random.uniform(1.0, 2.0, natoms)

        # Compute energy
        E1 = compute_gb_energy(coords, charges, radii)

        # Permute atoms 0 and 1
        perm = np.arange(natoms)
        perm[0], perm[1] = 1, 0

        coords_perm = coords[perm]
        charges_perm = charges[perm]
        radii_perm = radii[perm]

        E2 = compute_gb_energy(coords_perm, charges_perm, radii_perm)

        # Energies must be identical
        assert abs(E1 - E2) < 1e-10, f"Energy not symmetric: {E1} vs {E2}"

    @given(
        natoms=st.integers(min_value=2, max_value=20),
        scale_factor=st.floats(min_value=0.5, max_value=2.0)
    )
    def test_energy_scaling(self, natoms, scale_factor):
        """
        Property: GB energy must scale correctly with charge.

        E(λq) = λ² E(q) for uniform charge scaling
        """
        coords = np.random.uniform(-10, 10, (natoms, 3))
        charges = np.random.uniform(-1, 1, natoms)
        radii = np.random.uniform(1.0, 2.0, natoms)

        E1 = compute_gb_energy(coords, charges, radii)
        E2 = compute_gb_energy(coords, scale_factor * charges, radii)

        expected_ratio = scale_factor ** 2
        actual_ratio = E2 / (E1 + 1e-10)  # Avoid division by zero

        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"Energy scaling wrong: {actual_ratio} vs {expected_ratio}"

    @given(
        natoms=st.integers(min_value=2, max_value=10)
    )
    def test_force_energy_consistency(self, natoms):
        """
        Property: Forces must be consistent with energy gradient.

        F·dr should equal -dE for small displacements.
        """
        coords = np.random.uniform(-5, 5, (natoms, 3))
        charges = np.random.uniform(-1, 1, natoms)
        radii = np.random.uniform(1.0, 2.0, natoms)

        # Compute energy and forces at x
        E1, F1 = compute_gb_energy_forces(coords, charges, radii)

        # Small displacement
        dr = np.random.uniform(-0.01, 0.01, (natoms, 3))
        coords2 = coords + dr

        # Compute energy at x + dr
        E2, F2 = compute_gb_energy_forces(coords2, charges, radii)

        # Check consistency: -F·dr ≈ dE
        dE_numerical = E2 - E1
        dE_from_forces = -np.sum(F1 * dr)

        assert abs(dE_numerical - dE_from_forces) < 0.001, \
            f"Force-energy inconsistent: {dE_numerical} vs {dE_from_forces}"
```

---

## 3. Golden Reference Data Management

### 3.1 Data Storage Format

**Use HDF5 for efficient storage of numerical arrays**:

```
golden_references/
├── version_1.0/
│   ├── metadata.json
│   ├── small_systems.h5
│   ├── medium_systems.h5
│   └── large_systems.h5
│
└── version_1.1/
    └── ...
```

**HDF5 Structure**:
```
small_systems.h5
├── /water_dimer_2.8A/
│   ├── coords          [6, 3] float64
│   ├── charges         [6] float64
│   ├── born_radii      [6] float64
│   ├── energy          scalar float64
│   ├── forces          [6, 3] float64
│   └── metadata        {JSON string}
│
├── /two_oxygen_atoms/
│   └── ...
```

### 3.2 Reference Data Generator

```python
import h5py
import hashlib
from pathlib import Path

class GoldenReferenceManager:
    """Manage golden reference data for regression testing."""

    def __init__(self, version: str = "1.0"):
        self.version = version
        self.base_dir = Path("golden_references") / f"version_{version}"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def generate_golden_references(self, test_cases: List[TestCase]):
        """
        Generate golden reference values for all test cases.

        Uses highest-precision reference implementation.
        """
        small_file = h5py.File(self.base_dir / "small_systems.h5", 'w')
        medium_file = h5py.File(self.base_dir / "medium_systems.h5", 'w')
        large_file = h5py.File(self.base_dir / "large_systems.h5", 'w')

        for test in test_cases:
            # Select file based on system size
            natoms = test.system['natoms']
            if natoms <= 10:
                hf = small_file
            elif natoms <= 1000:
                hf = medium_file
            else:
                hf = large_file

            # Create group for this test case
            grp = hf.create_group(test.metadata['name'])

            # Store system data
            grp.create_dataset('coords', data=test.system['coordinates'])
            grp.create_dataset('charges', data=test.system['charges'])
            grp.create_dataset('atomic_numbers', data=test.system['atomic_numbers'])

            # Compute and store reference values
            if test.reference_values is None:
                refs = self._compute_reference_values(test)
            else:
                refs = test.reference_values

            grp.create_dataset('born_radii', data=refs['born_radii'])
            grp.create_dataset('energy', data=refs['energy'])
            if refs['forces'] is not None:
                grp.create_dataset('forces', data=refs['forces'])

            # Store metadata as JSON string
            import json
            grp.attrs['metadata'] = json.dumps(test.metadata)
            grp.attrs['parameters'] = json.dumps(test.parameters)
            grp.attrs['provenance'] = json.dumps(refs['provenance'])
            grp.attrs['tolerances'] = json.dumps(refs['tolerances'])

        small_file.close()
        medium_file.close()
        large_file.close()

        # Generate metadata file
        self._generate_metadata()

    def _compute_reference_values(self, test: TestCase) -> Dict:
        """Compute reference values using highest-precision method."""
        from fennol.validation.gb_reference import (
            GBReferenceOBC, GBEnergyReference, GBForcesNumerical
        )

        coords = np.array(test.system['coordinates'])
        charges = np.array(test.system['charges'])
        radii = np.array(test.parameters['radii'])
        b_params = np.array(test.parameters['b_params'])
        c_params = np.array(test.parameters['c_params'])

        # Born radii (NumPy reference)
        ref_born = GBReferenceOBC(cutoff=test.parameters['cutoff'])
        born_radii, psi_sum = ref_born.compute_born_radii(
            coords, radii, b_params, c_params
        )

        # Energy (NumPy reference)
        ref_energy = GBEnergyReference(dielectric=test.parameters['dielectric'])
        energy = ref_energy.compute_energy(coords, charges, born_radii)

        # Forces (numerical differentiation for small systems)
        if len(coords) <= 10:
            ref_forces = GBForcesNumerical(
                radii, b_params, c_params,
                test.parameters['dielectric'],
                test.parameters['cutoff']
            )
            forces = ref_forces.compute_forces_numerical(coords, charges)
        else:
            forces = None  # Too expensive for large systems

        return {
            'born_radii': born_radii,
            'psi_sum': psi_sum,
            'energy': energy,
            'forces': forces,
            'provenance': {
                'method': 'numpy_reference_fp64',
                'date': datetime.now().isoformat(),
                'code_version': self.version
            },
            'tolerances': self._get_tolerances(len(coords))
        }

    def _get_tolerances(self, natoms: int) -> Dict:
        """Get appropriate tolerances based on system size."""
        if natoms <= 10:
            return {
                'born_radii_abs': 1e-8,
                'born_radii_rel': 1e-6,
                'energy_abs': 1e-6,
                'energy_rel': 1e-5,
                'forces_abs': 1e-5,
                'forces_rel': 1e-3
            }
        elif natoms <= 1000:
            return {
                'born_radii_abs': 1e-6,
                'energy_abs': 1e-4,
                'forces_abs': 1e-3,
                'forces_rel': 1e-2
            }
        else:
            return {
                'born_radii_abs': 1e-5,
                'energy_abs': 1e-3,
                'forces_abs': 1e-2,
                'forces_rel': 5e-2
            }

    def _generate_metadata(self):
        """Generate metadata file for this reference dataset."""
        import git

        # Get git commit hash
        repo = git.Repo(search_parent_directories=True)
        commit_hash = repo.head.object.hexsha

        metadata = {
            'version': self.version,
            'date_generated': datetime.now().isoformat(),
            'git_commit': commit_hash,
            'reference_implementation': 'numpy_fp64',
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'description': 'Golden reference data for GB validation',
            'files': {
                'small_systems': 'small_systems.h5',
                'medium_systems': 'medium_systems.h5',
                'large_systems': 'large_systems.h5'
            }
        }

        with open(self.base_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_reference(self, test_name: str) -> Dict:
        """Load golden reference for a test case."""
        # Search all HDF5 files
        for hdf5_file in self.base_dir.glob("*.h5"):
            with h5py.File(hdf5_file, 'r') as hf:
                if test_name in hf:
                    grp = hf[test_name]
                    return {
                        'coords': grp['coords'][:],
                        'charges': grp['charges'][:],
                        'born_radii': grp['born_radii'][:],
                        'energy': grp['energy'][()],
                        'forces': grp['forces'][:] if 'forces' in grp else None,
                        'metadata': json.loads(grp.attrs['metadata']),
                        'tolerances': json.loads(grp.attrs['tolerances'])
                    }

        raise KeyError(f"Test case '{test_name}' not found in golden references")
```

### 3.3 Version Control Strategy

**Git + DVC (Data Version Control)**:

```bash
# Initialize DVC for large data files
dvc init

# Track golden reference files
dvc add golden_references/version_1.0/*.h5

# Commit to git
git add golden_references/version_1.0/.dvc
git add .dvc/config
git commit -m "Add golden references v1.0"

# Tag this version
git tag -a golden-v1.0 -m "Golden references version 1.0"
```

**Updating Golden References**:

```python
class ReferenceUpdateWorkflow:
    """Workflow for updating golden references when physics changes."""

    def validate_physics_change(
        self,
        old_version: str,
        new_version: str,
        expected_changes: List[str]
    ):
        """
        Validate that only expected components changed.

        Args:
            old_version: Previous reference version
            new_version: New reference version
            expected_changes: List of components expected to change
                             e.g., ['born_radii', 'energy']
        """
        old_mgr = GoldenReferenceManager(old_version)
        new_mgr = GoldenReferenceManager(new_version)

        # Load all test cases
        test_names = self._get_all_test_names(old_version)

        for test_name in test_names:
            old_ref = old_mgr.load_reference(test_name)
            new_ref = new_mgr.load_reference(test_name)

            # Check which components changed
            changes = {}
            if not np.allclose(old_ref['born_radii'], new_ref['born_radii']):
                changes['born_radii'] = self._compute_diff(
                    old_ref['born_radii'], new_ref['born_radii']
                )

            if not np.isclose(old_ref['energy'], new_ref['energy']):
                changes['energy'] = abs(
                    old_ref['energy'] - new_ref['energy']
                ) / abs(old_ref['energy'])

            if old_ref['forces'] is not None:
                if not np.allclose(old_ref['forces'], new_ref['forces']):
                    changes['forces'] = self._compute_diff(
                        old_ref['forces'], new_ref['forces']
                    )

            # Validate changes
            unexpected = set(changes.keys()) - set(expected_changes)
            if unexpected:
                raise ValueError(
                    f"Unexpected changes in {test_name}: {unexpected}\n"
                    f"Expected: {expected_changes}\n"
                    f"Got: {list(changes.keys())}"
                )

            # Log expected changes
            for component in expected_changes:
                if component in changes:
                    print(f"{test_name}/{component}: {changes[component]}")
```

---

## 4. Cross-Validation Strategy

### 4.1 Validation Matrix

```
            │ Analytical │ NumPy │ JAX │ CUDA  │ CUDA  │ OpenMM
            │ Solutions  │ FP64  │ AD  │ Basic │ Opt'd │
────────────┼────────────┼───────┼─────┼───────┼───────┼────────
Analytical  │     ─      │   ✓   │  ✓  │   ✓   │   ✓   │   ✓
────────────┼────────────┼───────┼─────┼───────┼───────┼────────
NumPy FP64  │     ✓      │   ─   │  ✓  │   ✓   │   ✓   │   ✓
────────────┼────────────┼───────┼─────┼───────┼───────┼────────
JAX AD      │     ✓      │   ✓   │  ─  │   ✓   │   ✓   │   ✓
────────────┼────────────┼───────┼─────┼───────┼───────┼────────
CUDA Basic  │     ✓      │   ✓   │  ✓  │   ─   │   ✓   │   ✓
────────────┼────────────┼───────┼─────┼───────┼───────┼────────
CUDA Opt'd  │     ✓      │   ✓   │  ✓  │   ✓   │   ─   │   ✓
────────────┼────────────┼───────┼─────┼───────┼───────┼────────
OpenMM      │     ✓      │   ✓   │  ✓  │   ✓   │   ✓   │   ─
```

### 4.2 Critical Validation Pairs

**Priority 1 (Blocking for release)**:
1. CUDA Optimized vs CUDA Baseline (same physics, different performance)
2. CUDA Baseline vs NumPy Reference (validate CUDA implementation)
3. Small systems: All implementations vs Analytical

**Priority 2 (Important for confidence)**:
4. JAX AutoDiff vs Numerical Finite Differences (force validation)
5. CUDA vs OpenMM (external validation)
6. Property-based tests (invariant validation)

**Priority 3 (Nice to have)**:
7. Energy conservation in MD (statistical validation)
8. Comparison with quantum chemistry (small molecules)

### 4.3 Tolerance Specifications

```python
class ValidationTolerances:
    """Tolerance specifications for cross-validation."""

    # Analytical vs Implementation
    ANALYTICAL_TOLERANCE = {
        'born_radii': {'abs': 1e-10, 'rel': 1e-8},
        'energy': {'abs': 1e-10, 'rel': 1e-8},
        'forces': {'abs': 1e-8, 'rel': 1e-6}
    }

    # NumPy Reference vs CUDA
    REFERENCE_TOLERANCE = {
        'born_radii': {'abs': 1e-6, 'rel': 1e-5},
        'energy': {'abs': 1e-5, 'rel': 1e-4},
        'forces': {'abs': 1e-4, 'rel': 1e-3}
    }

    # CUDA Baseline vs CUDA Optimized
    OPTIMIZATION_TOLERANCE = {
        'born_radii': {'abs': 1e-8, 'rel': 1e-7},
        'energy': {'abs': 1e-7, 'rel': 1e-6},
        'forces': {'abs': 1e-6, 'rel': 1e-5}
    }

    # JAX vs NumPy (different backends)
    BACKEND_TOLERANCE = {
        'born_radii': {'abs': 1e-6, 'rel': 1e-5},
        'energy': {'abs': 1e-5, 'rel': 1e-4},
        'forces': {'abs': 1e-4, 'rel': 1e-3}
    }

    # OpenMM cross-validation (different code entirely)
    OPENMM_TOLERANCE = {
        'born_radii': {'abs': 1e-4, 'rel': 1e-3},
        'energy': {'abs': 1e-3, 'rel': 1e-2},
        'forces': {'abs': 1e-2, 'rel': 5e-2}
    }

    # Numerical derivatives (finite difference noise)
    NUMERICAL_TOLERANCE = {
        'forces': {'abs': 1e-3, 'rel': 1e-2}
    }

    @staticmethod
    def check_tolerance(
        value1: float,
        value2: float,
        tolerance: Dict[str, float],
        name: str = "value"
    ) -> bool:
        """
        Check if two values agree within tolerance.

        Returns True if |v1 - v2| < abs_tol OR |v1 - v2|/|v1| < rel_tol
        """
        abs_diff = abs(value1 - value2)
        abs_tol = tolerance['abs']

        # Absolute tolerance check
        if abs_diff < abs_tol:
            return True

        # Relative tolerance check
        if abs(value1) > 1e-10:  # Avoid division by zero
            rel_diff = abs_diff / abs(value1)
            rel_tol = tolerance['rel']
            if rel_diff < rel_tol:
                return True

        # Failed both checks
        print(f"Tolerance check FAILED for {name}:")
        print(f"  Value 1: {value1}")
        print(f"  Value 2: {value2}")
        print(f"  Abs diff: {abs_diff} (tol: {abs_tol})")
        if abs(value1) > 1e-10:
            print(f"  Rel diff: {rel_diff} (tol: {rel_tol})")

        return False
```

### 4.4 Disagreement Resolution Protocol

```
When two implementations disagree:

1. Check tolerance level
   ├─> Within tolerance → PASS
   └─> Outside tolerance → Investigate

2. Identify which implementation to trust
   Analytical > NumPy FP64 > JAX > CUDA Baseline > CUDA Optimized

3. Debugging workflow:
   a) Test on smaller system (isolate issue)
   b) Test individual components (Born radii, energy, forces separately)
   c) Add diagnostic prints to both implementations
   d) Compare intermediate values (psi, f_GB, etc.)
   e) Check for numerical stability issues

4. Common causes:
   - Floating point precision differences (FP32 vs FP64)
   - Different cutoff handling
   - Atomic operation ordering (CUDA)
   - Boundary conditions (PBC vs no PBC)
   - Missing terms (self-energy, Born radii derivatives)

5. Resolution:
   - If reference is wrong → Fix reference, regenerate golden data
   - If CUDA is wrong → Fix CUDA, add regression test
   - If tolerance too tight → Relax tolerance with justification
   - If fundamental disagreement → Escalate to physics expert
```

### 4.5 OpenMM Cross-Validation

```python
class OpenMMCrossValidator:
    """Cross-validate against OpenMM GB implementation."""

    def __init__(self):
        try:
            import openmm
            import openmm.app as app
            self.has_openmm = True
        except ImportError:
            self.has_openmm = False
            print("Warning: OpenMM not available for cross-validation")

    def setup_openmm_system(
        self,
        coords: np.ndarray,
        charges: np.ndarray,
        atomic_numbers: np.ndarray,
        parameters: Dict
    ):
        """Create OpenMM system with GB/OBC."""
        import openmm
        import openmm.app as app
        import openmm.unit as unit

        # Create system
        system = openmm.System()

        # Add particles
        for mass in self._get_masses(atomic_numbers):
            system.addParticle(mass * unit.amu)

        # Add GB force
        gb_force = openmm.GBSAOBCForce()
        gb_force.setSolventDielectric(parameters['dielectric'])
        gb_force.setSoluteDielectric(1.0)
        gb_force.setCutoffDistance(parameters['cutoff'] * unit.angstrom)

        # Set atomic parameters
        radii = parameters['radii']
        for i, (q, r) in enumerate(zip(charges, radii)):
            gb_force.addParticle(
                q,
                r * unit.angstrom,
                0.8  # OBC scaling factor
            )

        system.addForce(gb_force)

        return system

    def compute_openmm_reference(
        self,
        coords: np.ndarray,
        charges: np.ndarray,
        atomic_numbers: np.ndarray,
        parameters: Dict
    ) -> Dict:
        """Compute GB energy and forces using OpenMM."""
        if not self.has_openmm:
            return None

        import openmm
        import openmm.unit as unit

        # Setup system
        system = self.setup_openmm_system(
            coords, charges, atomic_numbers, parameters
        )

        # Create integrator (not used, but required)
        integrator = openmm.VerletIntegrator(1.0 * unit.femtosecond)

        # Create context
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, integrator, platform)

        # Set positions
        context.setPositions(coords * unit.angstrom)

        # Get state with energy and forces
        state = context.getState(getEnergy=True, getForces=True)

        energy = state.getPotentialEnergy().value_in_unit(
            unit.kilocalorie_per_mole
        )
        forces = state.getForces(asNumpy=True).value_in_unit(
            unit.kilocalorie_per_mole / unit.angstrom
        )

        return {
            'energy': energy,
            'forces': forces
        }

    def validate_against_openmm(
        self,
        test_case: TestCase,
        cuda_results: Dict
    ) -> Dict:
        """
        Validate CUDA results against OpenMM.

        Returns:
            validation_report: Dict with pass/fail and diagnostics
        """
        openmm_results = self.compute_openmm_reference(
            test_case.system['coordinates'],
            test_case.system['charges'],
            test_case.system['atomic_numbers'],
            test_case.parameters
        )

        if openmm_results is None:
            return {'status': 'skipped', 'reason': 'OpenMM not available'}

        # Compare energy
        energy_pass = ValidationTolerances.check_tolerance(
            cuda_results['energy'],
            openmm_results['energy'],
            ValidationTolerances.OPENMM_TOLERANCE['energy'],
            name='energy'
        )

        # Compare forces
        forces_max_diff = np.max(np.abs(
            cuda_results['forces'] - openmm_results['forces']
        ))
        forces_pass = forces_max_diff < ValidationTolerances.OPENMM_TOLERANCE['forces']['abs']

        return {
            'status': 'pass' if (energy_pass and forces_pass) else 'fail',
            'energy': {
                'cuda': cuda_results['energy'],
                'openmm': openmm_results['energy'],
                'diff': abs(cuda_results['energy'] - openmm_results['energy']),
                'pass': energy_pass
            },
            'forces': {
                'max_diff': forces_max_diff,
                'pass': forces_pass
            }
        }
```

---

## 5. Automated Testing Framework

### 5.1 Test Suite Organization

```
tests/
├── test_gb_analytical.py          # Analytical solutions
├── test_gb_reference.py            # NumPy reference validation
├── test_gb_cuda_baseline.py        # CUDA baseline tests
├── test_gb_cuda_optimized.py       # CUDA optimized tests
├── test_gb_cross_validation.py     # Cross-implementation tests
├── test_gb_properties.py           # Property-based tests
├── test_gb_regression.py           # Regression tests
└── test_gb_md_integration.py       # MD integration tests
```

### 5.2 Pytest Framework

```python
# File: tests/test_gb_cuda_optimized.py

import pytest
import numpy as np
from pathlib import Path

from fennol.validation.gb_reference import GBReferenceOBC, GBEnergyReference
from fennol.validation.golden_reference import GoldenReferenceManager
from fennol.validation.tolerances import ValidationTolerances
from fennol import cuda as fennol_cuda

class TestGBCUDAOptimized:
    """Test suite for optimized CUDA GB kernels."""

    @pytest.fixture(scope="class")
    def golden_refs(self):
        """Load golden reference data."""
        mgr = GoldenReferenceManager(version="1.0")
        return mgr

    @pytest.fixture(scope="class")
    def cuda_available(self):
        """Check if CUDA is available."""
        try:
            import fennol.cuda
            return True
        except ImportError:
            return False

    @pytest.mark.analytical
    @pytest.mark.parametrize("test_name", [
        "single_oxygen_atom",
        "two_atoms_far_15.0",
        "two_atoms_far_20.0"
    ])
    def test_analytical_cases(self, golden_refs, test_name, cuda_available):
        """Test CUDA against analytical solutions."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Load golden reference
        ref = golden_refs.load_reference(test_name)

        # Run CUDA
        coords = ref['coords']
        charges = ref['charges']
        radii = ref['radii']
        b_params = ref['b_params']
        c_params = ref['c_params']

        # Compute Born radii
        born_radii_cuda, psi_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff=12.0
        )

        # Compute energy and forces
        energy_cuda, forces_cuda = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_radii_cuda, radii,
            b_params, c_params, psi_cuda,
            dielectric=80.0, cutoff=12.0
        )

        # Validate Born radii
        assert np.allclose(
            born_radii_cuda, ref['born_radii'],
            atol=ValidationTolerances.ANALYTICAL_TOLERANCE['born_radii']['abs'],
            rtol=ValidationTolerances.ANALYTICAL_TOLERANCE['born_radii']['rel']
        ), f"Born radii mismatch for {test_name}"

        # Validate energy
        assert np.isclose(
            energy_cuda[0], ref['energy'],
            atol=ValidationTolerances.ANALYTICAL_TOLERANCE['energy']['abs'],
            rtol=ValidationTolerances.ANALYTICAL_TOLERANCE['energy']['rel']
        ), f"Energy mismatch for {test_name}"

        # Validate forces
        if ref['forces'] is not None:
            assert np.allclose(
                forces_cuda, ref['forces'],
                atol=ValidationTolerances.ANALYTICAL_TOLERANCE['forces']['abs'],
                rtol=ValidationTolerances.ANALYTICAL_TOLERANCE['forces']['rel']
            ), f"Forces mismatch for {test_name}"

    @pytest.mark.reference
    @pytest.mark.parametrize("system_size", [3, 6, 12, 24])
    def test_vs_numpy_reference(self, system_size, cuda_available):
        """Test CUDA vs NumPy reference on small systems."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Generate random test case
        np.random.seed(42 + system_size)
        coords = np.random.uniform(-5, 5, (system_size, 3))
        charges = np.random.uniform(-1, 1, system_size)
        radii = np.full(system_size, 1.5)
        b_params = np.full(system_size, 0.8)
        c_params = np.full(system_size, 0.0)

        # NumPy reference
        ref_born = GBReferenceOBC(cutoff=12.0)
        ref_energy = GBEnergyReference(dielectric=80.0)

        born_radii_ref, psi_ref = ref_born.compute_born_radii(
            coords, radii, b_params, c_params
        )
        energy_ref = ref_energy.compute_energy(
            coords, charges, born_radii_ref
        )

        # CUDA
        born_radii_cuda, psi_cuda = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff=12.0
        )
        energy_cuda, forces_cuda = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_radii_cuda, radii,
            b_params, c_params, psi_cuda,
            dielectric=80.0, cutoff=12.0
        )

        # Validate
        assert np.allclose(
            born_radii_cuda, born_radii_ref,
            atol=ValidationTolerances.REFERENCE_TOLERANCE['born_radii']['abs'],
            rtol=ValidationTolerances.REFERENCE_TOLERANCE['born_radii']['rel']
        )

        assert np.isclose(
            energy_cuda[0], energy_ref,
            atol=ValidationTolerances.REFERENCE_TOLERANCE['energy']['abs'],
            rtol=ValidationTolerances.REFERENCE_TOLERANCE['energy']['rel']
        )

    @pytest.mark.optimization
    def test_cuda_optimized_vs_baseline(self, cuda_available):
        """Test that optimized CUDA gives same results as baseline."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Medium-size test case
        np.random.seed(100)
        coords = np.random.uniform(-10, 10, (100, 3))
        charges = np.random.uniform(-1, 1, 100)
        radii = np.full(100, 1.5)
        b_params = np.full(100, 0.8)
        c_params = np.full(100, 0.0)

        # Baseline (if available)
        try:
            born_baseline, psi_baseline = fennol_cuda.gb_compute_born_radii_basic(
                coords, radii, b_params, c_params, cutoff=12.0
            )
            energy_baseline, forces_baseline = fennol_cuda.gb_compute_energy_forces_basic(
                coords, charges, born_baseline, dielectric=80.0, cutoff=12.0
            )
        except AttributeError:
            pytest.skip("Baseline CUDA implementation not available")

        # Optimized
        born_opt, psi_opt = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff=12.0
        )
        energy_opt, forces_opt = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born_opt, radii,
            b_params, c_params, psi_opt,
            dielectric=80.0, cutoff=12.0
        )

        # Must match exactly (same physics!)
        assert np.allclose(
            born_opt, born_baseline,
            atol=ValidationTolerances.OPTIMIZATION_TOLERANCE['born_radii']['abs'],
            rtol=ValidationTolerances.OPTIMIZATION_TOLERANCE['born_radii']['rel']
        )

        assert np.isclose(
            energy_opt[0], energy_baseline[0],
            atol=ValidationTolerances.OPTIMIZATION_TOLERANCE['energy']['abs'],
            rtol=ValidationTolerances.OPTIMIZATION_TOLERANCE['energy']['rel']
        )

    @pytest.mark.regression
    @pytest.mark.parametrize("regression_case", [
        "issue_42_nan_forces",
        # Add more as they're discovered
    ])
    def test_regression_cases(self, golden_refs, regression_case, cuda_available):
        """Test known failure cases that have been fixed."""
        if not cuda_available:
            pytest.skip("CUDA not available")

        # Load regression test case
        ref = golden_refs.load_reference(regression_case)

        # Run CUDA (should not crash/NaN)
        born_radii, psi = fennol_cuda.gb_compute_born_radii_with_psi(
            ref['coords'], ref['radii'],
            ref['b_params'], ref['c_params'],
            cutoff=12.0
        )
        energy, forces = fennol_cuda.gb_compute_forces_complete(
            ref['coords'], ref['charges'], born_radii, ref['radii'],
            ref['b_params'], ref['c_params'], psi,
            dielectric=80.0, cutoff=12.0
        )

        # Check for NaN/Inf
        assert not np.any(np.isnan(born_radii)), "Born radii contains NaN"
        assert not np.any(np.isnan(forces)), "Forces contain NaN"
        assert not np.isnan(energy[0]), "Energy is NaN"

        assert not np.any(np.isinf(born_radii)), "Born radii contains Inf"
        assert not np.any(np.isinf(forces)), "Forces contain Inf"
        assert not np.isinf(energy[0]), "Energy is Inf"
```

### 5.3 Property-Based Testing with Hypothesis

```python
# File: tests/test_gb_properties.py

import pytest
import hypothesis
from hypothesis import given, strategies as st, settings
import hypothesis.extra.numpy as npst
import numpy as np

from fennol import cuda as fennol_cuda

class TestGBProperties:
    """Property-based tests for GB implementation."""

    @given(
        natoms=st.integers(min_value=2, max_value=20),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(deadline=None, max_examples=50)
    def test_energy_permutation_invariance(self, natoms, seed):
        """
        Property: Energy must be invariant under atom permutation.

        E(atoms [i, j, k, ...]) = E(atoms [j, i, k, ...])
        """
        np.random.seed(seed)

        # Generate random configuration
        coords = np.random.uniform(-5, 5, (natoms, 3))
        charges = np.random.uniform(-1, 1, natoms)
        radii = np.random.uniform(1.0, 2.0, natoms)
        b_params = np.full(natoms, 0.8)
        c_params = np.full(natoms, 0.0)

        # Compute energy
        born1, psi1 = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff=12.0
        )
        energy1, _ = fennol_cuda.gb_compute_energy_forces(
            coords, charges, born1, dielectric=80.0, cutoff=12.0
        )

        # Permute atoms
        perm = np.random.permutation(natoms)
        coords_perm = coords[perm]
        charges_perm = charges[perm]
        radii_perm = radii[perm]
        b_perm = b_params[perm]
        c_perm = c_params[perm]

        # Compute energy again
        born2, psi2 = fennol_cuda.gb_compute_born_radii_with_psi(
            coords_perm, radii_perm, b_perm, c_perm, cutoff=12.0
        )
        energy2, _ = fennol_cuda.gb_compute_energy_forces(
            coords_perm, charges_perm, born2, dielectric=80.0, cutoff=12.0
        )

        # Energies must be identical
        assert np.isclose(energy1[0], energy2[0], atol=1e-6), \
            f"Energy not permutation invariant: {energy1[0]} vs {energy2[0]}"

    @given(
        natoms=st.integers(min_value=2, max_value=10),
        scale=st.floats(min_value=0.1, max_value=3.0),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(deadline=None, max_examples=30)
    def test_energy_charge_scaling(self, natoms, scale, seed):
        """
        Property: E(λq) = λ² E(q) for uniform charge scaling.
        """
        np.random.seed(seed)

        coords = np.random.uniform(-5, 5, (natoms, 3))
        charges = np.random.uniform(-1, 1, natoms)
        radii = np.random.uniform(1.0, 2.0, natoms)
        b_params = np.full(natoms, 0.8)
        c_params = np.full(natoms, 0.0)

        # Compute with original charges
        born, psi = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff=12.0
        )
        energy1, _ = fennol_cuda.gb_compute_energy_forces(
            coords, charges, born, dielectric=80.0, cutoff=12.0
        )

        # Compute with scaled charges
        energy2, _ = fennol_cuda.gb_compute_energy_forces(
            coords, scale * charges, born, dielectric=80.0, cutoff=12.0
        )

        # Check scaling
        expected_ratio = scale ** 2
        if abs(energy1[0]) > 1e-6:
            actual_ratio = energy2[0] / energy1[0]
            assert np.isclose(actual_ratio, expected_ratio, rtol=0.01), \
                f"Energy scaling wrong: {actual_ratio} vs {expected_ratio}"

    @given(
        natoms=st.integers(min_value=2, max_value=8),
        seed=st.integers(min_value=0, max_value=10000)
    )
    @settings(deadline=None, max_examples=20)
    def test_force_energy_consistency(self, natoms, seed):
        """
        Property: F·dr = -dE for small displacements.

        This is a fundamental check that forces are energy gradients.
        """
        np.random.seed(seed)

        coords = np.random.uniform(-5, 5, (natoms, 3))
        charges = np.random.uniform(-1, 1, natoms)
        radii = np.random.uniform(1.0, 2.0, natoms)
        b_params = np.full(natoms, 0.8)
        c_params = np.full(natoms, 0.0)

        # Compute at x
        born1, psi1 = fennol_cuda.gb_compute_born_radii_with_psi(
            coords, radii, b_params, c_params, cutoff=12.0
        )
        energy1, forces1 = fennol_cuda.gb_compute_forces_complete(
            coords, charges, born1, radii, b_params, c_params, psi1,
            dielectric=80.0, cutoff=12.0
        )

        # Small random displacement
        dr = np.random.uniform(-0.01, 0.01, (natoms, 3))
        coords2 = coords + dr

        # Compute at x + dr
        born2, psi2 = fennol_cuda.gb_compute_born_radii_with_psi(
            coords2, radii, b_params, c_params, cutoff=12.0
        )
        energy2, _ = fennol_cuda.gb_compute_forces_complete(
            coords2, charges, born2, radii, b_params, c_params, psi2,
            dielectric=80.0, cutoff=12.0
        )

        # Check: -F·dr ≈ dE
        dE_numerical = energy2[0] - energy1[0]
        dE_from_forces = -np.sum(forces1 * dr)

        if abs(dE_numerical) > 1e-6:
            rel_error = abs(dE_numerical - dE_from_forces) / abs(dE_numerical)
            assert rel_error < 0.1, \
                f"Force-energy inconsistent: dE={dE_numerical}, F·dr={dE_from_forces}"
```

### 5.4 Continuous Integration Configuration

```yaml
# File: .github/workflows/gb_validation.yml

name: GB Validation Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-analytical:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov hypothesis
      - name: Run analytical tests
        run: pytest tests/test_gb_analytical.py -v --cov

  test-reference:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest numpy scipy
      - name: Run reference tests
        run: pytest tests/test_gb_reference.py -v

  test-cuda:
    runs-on: self-hosted  # Requires GPU
    steps:
      - uses: actions/checkout@v3
      - name: Set up CUDA environment
        run: |
          export CUDA_HOME=/usr/local/cuda
          export PATH=$CUDA_HOME/bin:$PATH
      - name: Build CUDA kernels
        run: |
          cd src/fennol/cuda
          cmake .
          make
      - name: Run CUDA tests
        run: |
          pytest tests/test_gb_cuda_optimized.py -v --tb=short

  test-property-based:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install hypothesis pytest
      - name: Run property-based tests
        run: |
          pytest tests/test_gb_properties.py -v --hypothesis-show-statistics

  validate-golden-references:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          lfs: true  # Pull large files from Git LFS
      - name: Verify golden reference integrity
        run: |
          python scripts/verify_golden_refs.py
```

---

## 6. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

**Deliverables**:
1. ✅ NumPy reference implementations
   - `GBReferenceOBC` class
   - `GBEnergyReference` class
   - `GBForcesNumerical` class
2. ✅ Test case generator
   - Analytical cases
   - Small systems (N < 20)
3. ✅ Basic validation framework
   - Tolerance specifications
   - Simple pytest tests

**Success Criteria**:
- All analytical test cases pass
- NumPy reference validated against hand calculations

### Phase 2: Test Case Library (Week 3-4)

**Deliverables**:
1. ✅ Complete test case library
   - 10 analytical cases
   - 50 small system cases
   - 20 medium system cases
   - 5 large system cases
2. ✅ Golden reference generation
   - HDF5 storage
   - Version control setup
3. ✅ Property-based test suite
   - Hypothesis integration
   - 10 property tests

**Success Criteria**:
- 85 test cases with golden references
- All property tests passing

### Phase 3: CUDA Validation (Week 5-6)

**Deliverables**:
1. ✅ CUDA baseline validation
   - Compare vs NumPy reference
   - Identify any discrepancies
2. ✅ CUDA optimization validation
   - Compare optimized vs baseline
   - Performance benchmarks
3. ✅ Cross-validation matrix
   - All critical pairs tested

**Success Criteria**:
- CUDA passes all reference tests
- Optimization preserves physics within tolerance

### Phase 4: Integration & Automation (Week 7-8)

**Deliverables**:
1. ✅ CI/CD integration
   - GitHub Actions workflows
   - Automated test runs
2. ✅ Regression test database
   - Known issues documented
   - Automated regression testing
3. ✅ Documentation
   - User guide for validation
   - Developer guide for adding tests

**Success Criteria**:
- All tests run automatically on PR
- 100% test coverage for GB kernels
- Documentation complete

### Phase 5: Advanced Validation (Week 9-10)

**Deliverables**:
1. ✅ OpenMM cross-validation
   - Side-by-side comparison
   - Detailed diagnostics
2. ✅ MD trajectory validation
   - Energy conservation tests
   - Statistical property tests
3. ✅ Performance regression tracking
   - Benchmark database
   - Automated performance tests

**Success Criteria**:
- Agreement with OpenMM within tolerance
- Energy conserved in MD
- 10× speedup achieved and validated

---

## Appendix A: File Organization

```
fennol/
├── src/fennol/
│   ├── validation/
│   │   ├── __init__.py
│   │   ├── gb_reference.py          # NumPy reference implementations
│   │   ├── test_generator.py        # Test case generator
│   │   ├── golden_reference.py      # Golden reference manager
│   │   ├── tolerances.py            # Tolerance specifications
│   │   ├── cross_validator.py       # Cross-validation tools
│   │   └── openmm_validator.py      # OpenMM cross-validation
│   │
│   └── cuda/
│       ├── src/
│       │   ├── gb_born_radii.cu     # Born radii kernels
│       │   ├── gb_energy_forces.cu   # Energy/force kernels
│       │   └── gb_forces_complete.cu # Complete force implementation
│       └── tests/
│           └── benchmark_gb.py       # Performance benchmarks
│
├── tests/
│   ├── test_gb_analytical.py
│   ├── test_gb_reference.py
│   ├── test_gb_cuda_baseline.py
│   ├── test_gb_cuda_optimized.py
│   ├── test_gb_cross_validation.py
│   ├── test_gb_properties.py
│   ├── test_gb_regression.py
│   └── test_gb_md_integration.py
│
├── test_cases/
│   ├── analytical/
│   ├── small/
│   ├── medium/
│   ├── large/
│   └── regression/
│
├── golden_references/
│   ├── version_1.0/
│   │   ├── metadata.json
│   │   ├── small_systems.h5
│   │   ├── medium_systems.h5
│   │   └── large_systems.h5
│   └── version_1.1/
│
├── docs/
│   ├── GB_VALIDATION_FRAMEWORK_DESIGN.md    # This document
│   ├── GB_VALIDATION_USER_GUIDE.md
│   └── GB_VALIDATION_DEVELOPER_GUIDE.md
│
└── scripts/
    ├── generate_test_cases.py
    ├── generate_golden_refs.py
    ├── verify_golden_refs.py
    └── run_full_validation.py
```

---

## Appendix B: Quick Start Guide

### For Users

**Run validation suite**:
```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest tests/test_gb_*.py -v

# Run specific category
pytest tests/test_gb_analytical.py -v
pytest tests/test_gb_cuda_optimized.py -v

# Run with coverage
pytest tests/ --cov=fennol.cuda --cov=fennol.validation
```

### For Developers

**Add new test case**:
```python
from fennol.validation.test_generator import TestCaseGenerator

gen = TestCaseGenerator(output_dir="test_cases/my_new_cases")
test = gen.generate_custom_case(...)
test.to_json("test_cases/my_new_cases/my_test.json")
```

**Validate new CUDA optimization**:
```python
from fennol.validation.cross_validator import CUDAValidator

validator = CUDAValidator()
report = validator.validate_optimization(
    test_case_name="water_box_64",
    baseline_func=fennol_cuda.gb_compute_basic,
    optimized_func=fennol_cuda.gb_compute_optimized
)
print(report)
```

**Update golden references**:
```bash
python scripts/generate_golden_refs.py --version 1.1
python scripts/verify_golden_refs.py --version 1.1

# Commit to git/DVC
git add golden_references/version_1.1/.dvc
git commit -m "Update golden refs to v1.1"
git tag golden-v1.1
```

---

**End of Document**
