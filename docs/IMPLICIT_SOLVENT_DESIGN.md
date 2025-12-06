# Implicit Solvent Model Design

## Overview

This document describes the design and implementation of GPU-native implicit solvent models for FeNNol. The framework is modular to allow easy addition of new solvent models over time.

## Motivation

**Benefits of implicit solvent**:
1. **Reduced system size**: Eliminate explicit water molecules (3x fewer atoms for solvated proteins)
2. **Lower memory requirements**: DHFR (23,558 atoms) → DHFR protein only (~2,500 atoms)
3. **Faster sampling**: Reduced degrees of freedom, no water viscosity
4. **GPU feasibility**: Systems that exceed GPU memory with explicit solvent become feasible

**Trade-offs**:
- Less accurate than explicit solvent for some properties
- No explicit water-mediated effects
- Requires careful parametrization

## Architecture

### Modular Design

```
src/fennol/models/physics/implicit_solvent/
├── __init__.py
├── base.py              # Base class for all implicit solvent models
├── generalized_born.py  # GB/SA model (initial implementation)
├── poisson_boltzmann.py # PB model (future)
├── cosmo.py             # COSMO model (future)
└── parameters.py        # Atomic radii and other parameters
```

### CUDA Kernels

```
src/fennol/cuda/
├── include/
│   └── implicit_solvent.cuh   # Common headers
├── src/
│   ├── gb_pairwise.cu         # Pairwise GB calculations
│   ├── gb_born_radii.cu       # Born radii computation
│   └── gb_forces.cu           # Force calculations
└── src/bindings.cpp           # Python bindings
```

## Generalized Born (GB) Model

### Theory

The GB model approximates the Poisson-Boltzmann solvation free energy:

**Total solvation energy**:
```
ΔG_solv = ΔG_elec + ΔG_nonpolar
```

**Electrostatic component** (Born model):
```
ΔG_elec = -½ (1 - 1/ε) Σᵢⱼ qᵢqⱼ / f_GB(rᵢⱼ, Rᵢ, Rⱼ)
```

Where:
- `ε` = solvent dielectric constant (80 for water)
- `qᵢ, qⱼ` = partial atomic charges
- `rᵢⱼ` = distance between atoms i and j
- `Rᵢ, Rⱼ` = Born radii (effective solvation radii)
- `f_GB` = generalized Born function

**GB function** (Still et al. 1990):
```
f_GB(rᵢⱼ, Rᵢ, Rⱼ) = √(rᵢⱼ² + RᵢRⱼ exp(-rᵢⱼ²/4RᵢRⱼ))
```

**Born radii** (OBC model, Onufriev et al. 2004):
```
1/Rᵢ = 1/ρᵢ - tanh(ψᵢ - bᵢψᵢ² + cᵢψᵢ³) / ρᵢ
```

Where:
- `ρᵢ` = intrinsic atomic radius
- `ψᵢ` = pairwise descreening integral
- `bᵢ, cᵢ` = empirical parameters (element-specific)

**Non-polar component** (surface area):
```
ΔG_nonpolar = Σᵢ γᵢ SAᵢ
```

Where:
- `γᵢ` = surface tension coefficient (element-specific)
- `SAᵢ` = solvent-accessible surface area of atom i

### Implementation Variants

1. **GB/SA** (Generalized Born + Surface Area)
   - Original Still et al. formulation
   - Good for proteins
   - Fast, well-tested

2. **OBC (Onufriev-Bashford-Case)**
   - Improved Born radii calculation
   - Better for nucleic acids
   - Parameters: α=0.8, β=0, γ=2.909125

3. **GBn** (GB-neck models, Nguyen et al.)
   - Accounts for "neck" regions between atoms
   - More accurate for tight packing
   - Slightly more expensive

**Initial implementation**: OBC (GB/SA with OBC radii)

### Algorithm

**Step 1: Compute Born radii**
```
For each atom i:
  ψᵢ = 0
  For each atom j ≠ i within cutoff:
    rᵢⱼ = |rᵢ - rⱼ|
    Compute pairwise descreening integral
    ψᵢ += I(rᵢⱼ, ρᵢ, ρⱼ)

  Rᵢ = compute_born_radius(ρᵢ, ψᵢ, bᵢ, cᵢ)
```

**Step 2: Compute GB electrostatic energy and forces**
```
ΔG_elec = 0
For each pair (i,j):
  f_GB = sqrt(rᵢⱼ² + RᵢRⱼ exp(-rᵢⱼ²/4RᵢRⱼ))
  ΔG_elec += qᵢqⱼ / f_GB

Forces from chain rule: dE/drᵢ = ∂E/∂rᵢⱼ + Σⱼ ∂E/∂Rⱼ ∂Rⱼ/∂rᵢⱼ
```

**Step 3: Compute non-polar energy and forces**
```
For each atom i:
  SAᵢ = compute_accessible_surface_area(i)
  ΔG_nonpolar += γᵢ SAᵢ
```

### Parameters

**Atomic radii** (Bondi radii + OBC parameters):

| Element | ρ (Å) | b | c | γ (kcal/mol/Ų) |
|---------|-------|---|---|----------------|
| H | 1.20 | 0.85 | 0.72 | 0.005 |
| C | 1.70 | 0.72 | -0.01 | 0.005 |
| N | 1.55 | 0.79 | 0.28 | 0.005 |
| O | 1.50 | 0.85 | 0.10 | 0.005 |
| S | 1.80 | 0.96 | -0.02 | 0.005 |
| P | 1.85 | 0.86 | 0.00 | 0.005 |

**Solvent parameters**:
- Dielectric constant (water): ε = 80.0
- Solvent probe radius: 1.4 Å
- Cutoff distance: 12-15 Å (same as non-bonded)

## CUDA Implementation

### Kernel Design

**1. Born Radii Kernel** (`compute_born_radii.cu`):
```cuda
__global__ void compute_pairwise_descreening(
    int natoms,
    const double* coords,      // [natoms, 3]
    const double* radii,       // [natoms]
    const double* b_params,    // [natoms]
    const double* c_params,    // [natoms]
    const int* neighborlist,   // [natoms, max_neighbors]
    const int* n_neighbors,    // [natoms]
    double* born_radii         // [natoms] - output
)
```

**2. GB Energy/Force Kernel** (`gb_pairwise.cu`):
```cuda
__global__ void compute_gb_energy_forces(
    int natoms,
    const double* coords,      // [natoms, 3]
    const double* charges,     // [natoms]
    const double* born_radii,  // [natoms]
    const int* neighborlist,
    const int* n_neighbors,
    double epsilon,            // solvent dielectric
    double* energy,            // [1] - output
    double* forces             // [natoms, 3] - output
)
```

**3. Surface Area Kernel** (`gb_surface_area.cu`):
```cuda
__global__ void compute_sasa_energy_forces(
    int natoms,
    const double* coords,
    const double* radii,
    const double* gamma,       // surface tension
    double probe_radius,
    double* energy,
    double* forces
)
```

### Optimization Strategies

1. **Neighborlist reuse**: Use existing MD neighborlist for GB calculations
2. **Born radii caching**: Only recompute when neighborlist updates
3. **Atomic operations**: Use for force accumulation
4. **Shared memory**: Cache coordinates/radii for thread block
5. **Warp-level primitives**: Use warp shuffle for reductions

### Memory Layout

**Per-atom data**:
```cpp
struct AtomData {
    double3 coords;        // Position
    double charge;         // Partial charge
    double radius;         // Intrinsic radius
    double born_radius;    // Effective Born radius
    double b_param;        // OBC b parameter
    double c_param;        // OBC c parameter
    double gamma;          // Surface tension coefficient
};
```

## Integration with FeNNol

### Configuration

**Input file** (`input.fnl`):
```fnl
# Enable implicit solvent
implicit_solvent {
  model = "OBC"              # GB/SA, OBC, GBn, PB, COSMO
  dielectric = 80.0          # Solvent dielectric constant
  cutoff = 12.0              # GB cutoff distance (Å)

  # Non-polar component
  surface_tension = 0.005    # kcal/mol/Ų
  probe_radius = 1.4         # Solvent probe radius (Å)

  # Model-specific parameters
  obc_alpha = 0.8
  obc_beta = 0.0
  obc_gamma = 2.909125
}
```

### Code Structure

**Base class** (`base.py`):
```python
from abc import ABC, abstractmethod
from typing import Dict, Tuple
import jax.numpy as jnp

class ImplicitSolventModel(ABC):
    """Base class for implicit solvent models."""

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    @abstractmethod
    def compute_energy_forces(
        self,
        coords: jnp.ndarray,
        charges: jnp.ndarray,
        atomic_numbers: jnp.ndarray,
        box: jnp.ndarray = None
    ) -> Tuple[float, jnp.ndarray]:
        """
        Compute solvation energy and forces.

        Args:
            coords: Atomic coordinates [natoms, 3]
            charges: Partial atomic charges [natoms]
            atomic_numbers: Atomic numbers [natoms]
            box: Simulation box (optional for PBC)

        Returns:
            energy: Solvation free energy (kcal/mol)
            forces: Forces on atoms [natoms, 3] (kcal/mol/Å)
        """
        pass
```

**GB implementation** (`generalized_born.py`):
```python
class GeneralizedBorn(ImplicitSolventModel):
    """OBC Generalized Born implicit solvent model."""

    def __init__(self, parameters: Dict):
        super().__init__(parameters)
        self.variant = parameters.get("variant", "OBC")
        self.dielectric = parameters.get("dielectric", 80.0)
        self.cutoff = parameters.get("cutoff", 12.0)
        # Load atomic parameters...

    def compute_energy_forces(self, coords, charges, atomic_numbers, box=None):
        # Use CUDA kernels or JAX implementation
        if has_cuda:
            return self._compute_cuda(coords, charges, atomic_numbers)
        else:
            return self._compute_jax(coords, charges, atomic_numbers)
```

### MD Integration

In `integrate.py`:
```python
# After computing MM/QM forces
if implicit_solvent is not None:
    solv_energy, solv_forces = implicit_solvent.compute_energy_forces(
        coords=system["coordinates"],
        charges=system["charges"],
        atomic_numbers=system["atomic_numbers"]
    )

    new_sys["epot"] = new_sys["epot"] + solv_energy
    new_sys["forces"] = new_sys["forces"] + solv_forces
    new_sys["solvation_energy"] = solv_energy
```

## Testing and Validation

### Test Systems

1. **Small molecules** (ion solvation):
   - Na⁺, Cl⁻, Ca²⁺ (compare to Born model)
   - Simple alcohols, amides

2. **Peptides**:
   - Alanine dipeptide
   - β-hairpin (Trp-cage)

3. **Proteins**:
   - DHFR (2,500 atoms without water)
   - Lysozyme

### Validation Metrics

1. **Energy conservation**: NVE ensemble drift
2. **Solvation free energies**: Compare to experimental ΔG_solv
3. **Structure**: RMSD vs explicit solvent simulations
4. **Performance**: Speed vs explicit solvent

### Reference Data

- **Experimental solvation energies**: Freesol v database
- **QM/MM**: MP2/aug-cc-pVTZ + PCM
- **Explicit solvent**: TIP3P 100 ns MD

## Future Extensions

### Additional Models

1. **Poisson-Boltzmann (PB)**:
   - More accurate than GB
   - Solves PB equation on grid
   - Slower but better for charged systems

2. **COSMO** (Conductor-like Screening Model):
   - GAMESS implementation
   - Good for QM/MM
   - Fast, segment-based

3. **SMD** (Solvation Model based on Density):
   - Universal solvent model
   - Works with any QM method
   - Parametrized for many solvents

### Advanced Features

1. **Salt effects**: Debye-Hückel screening
2. **pH-dependent protonation**: Constant-pH MD
3. **Membrane models**: Heterogeneous dielectric
4. **Mixed solvents**: Water/ethanol, etc.

## Performance Estimates

**DHFR example** (2,500 atoms without water):

| Component | Explicit Water | Implicit Solvent | Speedup |
|-----------|----------------|------------------|---------|
| Atoms | 23,558 | 2,500 | 9.4x fewer |
| Memory (est.) | >12 GB | ~1.5 GB | 8x less |
| Time/step (est.) | 2.78 s (CPU) | ~0.05 s (CUDA) | ~55x faster |

**Expected performance**:
- GB calculation: ~1-2 ms for 2,500 atoms on GPU
- Total speedup: 50-100x vs explicit solvent on CPU

## References

1. Still et al. (1990) - Original GB model
2. Onufriev et al. (2004) - OBC parameters
3. Nguyen et al. (2013) - GBn models
4. GAMESS documentation - COSMO/PCM
5. AMBER manual - GB/SA implementation

## Implementation Timeline

**Phase 1** (Initial implementation):
- [x] Design document
- [ ] Base class and framework
- [ ] OBC GB model (JAX)
- [ ] Parameter files
- [ ] Integration with MD
- [ ] Basic testing

**Phase 2** (GPU acceleration):
- [ ] CUDA kernels for Born radii
- [ ] CUDA kernels for GB energy/forces
- [ ] CUDA kernels for SASA
- [ ] Performance optimization
- [ ] Validation suite

**Phase 3** (Additional models):
- [ ] Poisson-Boltzmann
- [ ] COSMO
- [ ] SMD
- [ ] Advanced features

## Contact

For questions or contributions, please open an issue on GitHub.
