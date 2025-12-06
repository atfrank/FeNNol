"""
Tolerance Specifications for GB Validation

Defines precision requirements for validating optimizations at different
trust levels. Tighter tolerances for higher-trust comparisons.

Trust hierarchy (highest to lowest):
1. Analytical solutions (hand-calculated)
2. NumPy FP64 reference (slow, obviously correct)
3. JAX autodiff (medium speed, autodiff forces)
4. CUDA baseline (fast, unoptimized)
5. CUDA optimized (fastest, target)

Usage:
    from fennol.validation.tolerances import ValidationTolerances

    # Check if values agree within tolerance
    passed = ValidationTolerances.check_tolerance(
        value1, value2,
        ValidationTolerances.REFERENCE_TOLERANCE['energy'],
        name='energy'
    )
"""

import numpy as np
from typing import Dict


class ValidationTolerances:
    """Tolerance specifications for cross-validation."""

    # Analytical vs Implementation: VERY TIGHT
    # These should match to machine precision
    ANALYTICAL_TOLERANCE = {
        'born_radii': {'abs': 1e-10, 'rel': 1e-8},
        'energy': {'abs': 1e-10, 'rel': 1e-8},
        'forces': {'abs': 1e-8, 'rel': 1e-6}
    }

    # NumPy Reference vs CUDA: MEDIUM
    # Both are FP64, but different algorithms → some numerical differences
    REFERENCE_TOLERANCE = {
        'born_radii': {'abs': 1e-6, 'rel': 1e-5},
        'energy': {'abs': 1e-5, 'rel': 1e-4},
        'forces': {'abs': 1e-4, 'rel': 1e-3}
    }

    # CUDA Baseline vs CUDA Optimized: TIGHT
    # Same physics, must match closely (only FP rounding differences allowed)
    OPTIMIZATION_TOLERANCE = {
        'born_radii': {'abs': 1e-8, 'rel': 1e-7},
        'energy': {'abs': 1e-7, 'rel': 1e-6},
        'forces': {'abs': 1e-6, 'rel': 1e-5}
    }

    # JAX vs NumPy (different backends): MEDIUM
    BACKEND_TOLERANCE = {
        'born_radii': {'abs': 1e-6, 'rel': 1e-5},
        'energy': {'abs': 1e-5, 'rel': 1e-4},
        'forces': {'abs': 1e-4, 'rel': 1e-3}
    }

    # OpenMM cross-validation (different code entirely): LOOSE
    # Different choices (cutoff switching, parameter sets) → larger differences
    OPENMM_TOLERANCE = {
        'born_radii': {'abs': 1e-4, 'rel': 1e-3},
        'energy': {'abs': 1e-3, 'rel': 1e-2},
        'forces': {'abs': 1e-2, 'rel': 5e-2}
    }

    # Numerical derivatives (finite difference noise): LOOSE
    NUMERICAL_TOLERANCE = {
        'forces': {'abs': 1e-3, 'rel': 1e-2}
    }

    # MD trajectory validation
    MD_TOLERANCE = {
        'energy_drift_per_ps': 1e-4,  # 0.01% per ps
        'energy_drift_per_10000steps': 1e-3,  # 0.1% over 10k steps
        'temperature_deviation': 5.0,  # ±5 K from target
        'force_magnitude_max': 1000.0,  # kcal/(mol·Å) - sanity check
        'displacement_per_step_max': 0.1,  # Å for 1 fs timestep
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

        Uses both absolute AND relative tolerance:
        - Pass if |v1 - v2| < abs_tol
        - OR if |v1 - v2|/|v1| < rel_tol

        Args:
            value1: First value
            value2: Second value
            tolerance: Dict with 'abs' and 'rel' keys
            name: Name of value (for error messages)

        Returns:
            True if values agree within tolerance

        Example:
            >>> tol = ValidationTolerances.REFERENCE_TOLERANCE['energy']
            >>> ValidationTolerances.check_tolerance(100.0, 100.001, tol, 'energy')
            True
        """
        abs_diff = abs(value1 - value2)
        abs_tol = tolerance['abs']

        # Absolute tolerance check
        if abs_diff < abs_tol:
            return True

        # Relative tolerance check (avoid division by zero)
        if abs(value1) > 1e-10:
            rel_diff = abs_diff / abs(value1)
            rel_tol = tolerance['rel']
            if rel_diff < rel_tol:
                return True

        # Failed both checks - print diagnostic
        print(f"❌ Tolerance check FAILED for {name}:")
        print(f"  Value 1: {value1:.10e}")
        print(f"  Value 2: {value2:.10e}")
        print(f"  Abs diff: {abs_diff:.10e} (tol: {abs_tol:.10e})")
        if abs(value1) > 1e-10:
            print(f"  Rel diff: {rel_diff:.10e} (tol: {rel_tol:.10e})")

        return False

    @staticmethod
    def check_array_tolerance(
        array1: np.ndarray,
        array2: np.ndarray,
        tolerance: Dict[str, float],
        name: str = "array"
    ) -> bool:
        """
        Check if two arrays agree element-wise within tolerance.

        Args:
            array1: First array
            array2: Second array
            tolerance: Dict with 'abs' and 'rel' keys
            name: Name of array (for error messages)

        Returns:
            True if all elements agree within tolerance

        Example:
            >>> tol = ValidationTolerances.REFERENCE_TOLERANCE['forces']
            >>> forces1 = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> forces2 = np.array([[1.0001, 2.0001], [3.0001, 4.0001]])
            >>> ValidationTolerances.check_array_tolerance(
            ...     forces1, forces2, tol, 'forces'
            ... )
            True
        """
        if array1.shape != array2.shape:
            print(f"❌ Shape mismatch for {name}: {array1.shape} vs {array2.shape}")
            return False

        # Try np.allclose first (faster)
        if np.allclose(array1, array2, atol=tolerance['abs'], rtol=tolerance['rel']):
            return True

        # Find which elements failed
        abs_diff = np.abs(array1 - array2)
        max_abs_diff = np.max(abs_diff)
        max_abs_idx = np.unravel_index(np.argmax(abs_diff), abs_diff.shape)

        # Check relative for non-zero elements
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = abs_diff / np.abs(array1)
            rel_diff = np.where(np.abs(array1) < 1e-10, 0, rel_diff)
        max_rel_diff = np.max(rel_diff)
        max_rel_idx = np.unravel_index(np.argmax(rel_diff), rel_diff.shape)

        print(f"❌ Array tolerance check FAILED for {name}:")
        print(f"  Shape: {array1.shape}")
        print(f"  Max abs diff: {max_abs_diff:.10e} at {max_abs_idx} (tol: {tolerance['abs']:.10e})")
        print(f"  Max rel diff: {max_rel_diff:.10e} at {max_rel_idx} (tol: {tolerance['rel']:.10e})")
        print(f"  Value 1 at max: {array1[max_abs_idx]:.10e}")
        print(f"  Value 2 at max: {array2[max_abs_idx]:.10e}")

        return False

    @staticmethod
    def get_tolerances_for_size(natoms: int) -> Dict:
        """
        Get appropriate tolerances based on system size.

        Larger systems accumulate more numerical error, so we relax tolerances.

        Args:
            natoms: Number of atoms

        Returns:
            Dict of tolerances for this system size

        Example:
            >>> tols = ValidationTolerances.get_tolerances_for_size(1000)
            >>> print(tols['energy_abs'])
            1e-04
        """
        if natoms <= 10:
            # Small systems: very tight tolerances
            return {
                'born_radii_abs': 1e-8,
                'born_radii_rel': 1e-6,
                'energy_abs': 1e-6,
                'energy_rel': 1e-5,
                'forces_abs': 1e-5,
                'forces_rel': 1e-3
            }
        elif natoms <= 1000:
            # Medium systems: medium tolerances
            return {
                'born_radii_abs': 1e-6,
                'energy_abs': 1e-4,
                'forces_abs': 1e-3,
                'forces_rel': 1e-2
            }
        else:
            # Large systems: loose tolerances
            return {
                'born_radii_abs': 1e-5,
                'energy_abs': 1e-3,
                'forces_abs': 1e-2,
                'forces_rel': 5e-2
            }

    @staticmethod
    def check_no_nan_or_inf(
        *arrays: np.ndarray,
        names: list = None
    ) -> bool:
        """
        Check that arrays contain no NaN or Inf values.

        Args:
            *arrays: Variable number of numpy arrays to check
            names: Optional names for each array (for error messages)

        Returns:
            True if all arrays are finite

        Example:
            >>> forces = np.array([[1.0, 2.0], [3.0, 4.0]])
            >>> energy = np.array([100.0])
            >>> ValidationTolerances.check_no_nan_or_inf(
            ...     forces, energy, names=['forces', 'energy']
            ... )
            True
        """
        if names is None:
            names = [f"array_{i}" for i in range(len(arrays))]

        all_finite = True

        for arr, name in zip(arrays, names):
            if np.any(np.isnan(arr)):
                print(f"❌ {name} contains NaN values!")
                nan_indices = np.where(np.isnan(arr))
                print(f"  NaN at indices: {list(zip(*nan_indices))[:10]}")  # Show first 10
                all_finite = False

            if np.any(np.isinf(arr)):
                print(f"❌ {name} contains Inf values!")
                inf_indices = np.where(np.isinf(arr))
                print(f"  Inf at indices: {list(zip(*inf_indices))[:10]}")  # Show first 10
                all_finite = False

        return all_finite

    @staticmethod
    def check_newton_third_law(
        forces: np.ndarray,
        tolerance: float = 1e-8
    ) -> bool:
        """
        Check that net force is zero (momentum conservation).

        Args:
            forces: Force array [N, 3] in kcal/(mol·Å)
            tolerance: Tolerance for net force magnitude

        Returns:
            True if net force < tolerance

        Example:
            >>> forces = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
            >>> ValidationTolerances.check_newton_third_law(forces)
            True
        """
        net_force = np.sum(forces, axis=0)
        net_force_mag = np.linalg.norm(net_force)

        if net_force_mag < tolerance:
            return True

        print(f"❌ Newton's 3rd law check FAILED:")
        print(f"  Net force: {net_force}")
        print(f"  Magnitude: {net_force_mag:.10e} (tol: {tolerance:.10e})")

        return False
