# Simulated Annealing in FeNNol

## Overview

Simulated annealing (SA) is a probabilistic optimization technique that mimics the physical process of annealing in metallurgy. In molecular dynamics, simulated annealing is used to find low-energy conformations by:

1. **Heating**: Starting at high temperature to explore configuration space
2. **Cooling**: Gradually lowering temperature to settle into stable structures
3. **Equilibration**: Maintaining final temperature to verify stability

## Applications

Simulated annealing is widely used for:

- **Conformer search**: Finding global energy minimum of small molecules
- **Protein structure refinement**: Improving homology models or NMR structures
- **Docking pose optimization**: Refining ligand-protein binding modes
- **Crystal structure prediction**: Finding stable packing arrangements
- **Transition state search**: Locating saddle points on potential energy surfaces

## Quick Start

### Basic Example

```fnl
thermostat ANNEAL

temperature = 300.      # Base temperature (required)
gamma[THz] = 10.        # Langevin friction

annealing{
  T_start = 800.0       # Starting temperature (K)
  T_end = 50.0          # Ending temperature (K)
  schedule = exponential # Cooling schedule
  anneal_steps = 1.0    # Fraction of simulation to anneal
}
```

### Running Simulated Annealing

```bash
# Run with fennol_md (if installed)
fennol_md input_anneal.fnl

# Or run directly with Python
python -m fennol.md.run input_anneal.fnl
```

## Parameters

### Required Parameters

| Parameter | Description | Units | Example |
|-----------|-------------|-------|---------|
| `thermostat` | Must be set to `ANNEAL` or `ANNEALING` | - | `ANNEAL` |
| `temperature` | Base temperature (overridden by schedule) | K | `300.` |
| `gamma[THz]` | Langevin friction coefficient | THz | `10.` |

### Annealing Block Parameters

#### Temperature Range

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `T_start` | Starting temperature | Required | `800.0` |
| `T_end` | Final temperature | `0.0` | `50.0` |

Alternative (legacy) interface:
- `init_factor`: Starting temperature as factor of base temperature (default: `2.0`)
- `final_factor`: Final temperature as factor of base temperature (default: `0.01`)

#### Cooling Schedule

| Parameter | Values | Description |
|-----------|--------|-------------|
| `schedule` | `exponential` (default) | Exponential decay: T(t) = T_end + (T_start - T_end) × exp(-t/τ) |
| | `linear` | Linear decay: T(t) = T_start × (1 - t/t_max) + T_end × (t/t_max) |
| | `cosine` | Cosine decay: T(t) = T_end + 0.5×(T_start - T_end)×(1 + cos(πt/t_max)) |
| | `cosine_onecycle` | Heat up then cool down (legacy, for backward compatibility) |
| | `linear_onecycle` | Linear version of onecycle |

#### Timing

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| `anneal_steps` | Fraction of simulation to perform annealing | (0, 1] | `1.0` |

**Example**: If `nsteps = 100000` and `anneal_steps = 0.75`:
- Steps 0-74,999: Temperature decreases from T_start to T_end
- Steps 75,000-99,999: Temperature held constant at T_end (equilibration)

## Cooling Schedules Explained

### 1. Exponential Decay (Recommended)

**When to use**: Most general-purpose simulated annealing

**Characteristics**:
- Fast initial cooling, gradual approach to final temperature
- Mathematical form: `T(t) = T_end + (T_start - T_end) × exp(-5t/t_max)`
- Reaches ~99.3% of final temperature at t = t_max
- Most commonly used in literature

**Example**:
```fnl
annealing{
  T_start = 800.0
  T_end = 50.0
  schedule = exponential
  anneal_steps = 1.0
}
```

**Temperature profile**:
- 0% progress: 800 K
- 20% progress: ~420 K
- 50% progress: ~115 K
- 80% progress: ~65 K
- 100% progress: ~50 K

### 2. Linear Decay

**When to use**: Systematic exploration with constant cooling rate

**Characteristics**:
- Uniform cooling rate: dT/dt = constant
- Simple and predictable
- Good for debugging and benchmarking

**Example**:
```fnl
annealing{
  T_start = 600.0
  T_end = 100.0
  schedule = linear
  anneal_steps = 0.8
}
```

**Temperature profile**:
- 0% progress: 600 K
- 20% progress: 500 K
- 50% progress: 350 K
- 80% progress: 200 K (annealing ends)
- 100% progress: 100 K (equilibration)

### 3. Cosine Decay

**When to use**: Gentle annealing for sensitive systems (proteins, large molecules)

**Characteristics**:
- Slowest cooling initially (allows thorough exploration)
- Smooth S-curve profile
- Gentle approach to final temperature
- Best for maintaining protein secondary structure

**Example**:
```fnl
annealing{
  T_start = 500.0
  T_end = 300.0
  schedule = cosine
  anneal_steps = 1.0
}
```

**Temperature profile**:
- 0% progress: 500 K
- 20% progress: 481 K (slow initial cooling)
- 50% progress: 400 K (midpoint)
- 80% progress: 319 K (faster cooling)
- 100% progress: 300 K

## Best Practices

### 1. Choosing Temperature Range

| System Type | T_start | T_end | Rationale |
|-------------|---------|-------|-----------|
| Small molecules | 800-1200 K | 0-50 K | High T overcomes rotation barriers |
| Peptides (< 20 residues) | 500-700 K | 100-300 K | Moderate T explores conformations |
| Proteins | 350-450 K | 300 K | Conservative to avoid denaturation |
| Crystals/solids | 400-600 K | 100-200 K | Below melting point |

**Rule of thumb**: T_start should be ~2-3× the temperature where system becomes fluid, T_end should be your target equilibrium temperature.

### 2. Choosing Annealing Duration

| Cooling Rate | Steps | Use Case |
|-------------|-------|----------|
| Fast | 10,000-50,000 | Quick conformer search |
| Medium | 50,000-200,000 | Standard protein refinement |
| Slow | 200,000-1,000,000 | Careful optimization, large systems |

**Rule of thumb**: Slower cooling → better convergence to global minimum, but higher computational cost.

### 3. Multiple Independent Runs

For finding global minimum:
```bash
# Run 10-20 independent annealing simulations
for i in {1..20}; do
    fennol_md input_anneal.fnl --seed $i --output traj_$i.xyz
done

# Analyze final energies to find best conformation
```

### 4. Two-Stage Protocol

For difficult optimization problems:

**Stage 1: Aggressive annealing**
```fnl
nsteps = 100000
annealing{
  T_start = 1000.0
  T_end = 50.0
  schedule = exponential
}
```

**Stage 2: Refinement**
```fnl
nsteps = 50000
annealing{
  T_start = 300.0
  T_end = 100.0
  schedule = cosine
}
```

## Common Use Cases

### Use Case 1: Conformer Search (Small Molecule)

**Goal**: Find global energy minimum of aspirin

```fnl
nsteps = 50000
dt[fs] = 0.5

thermostat ANNEAL
temperature = 300.
gamma[THz] = 10.

annealing{
  T_start = 1000.0    # High T to overcome barriers
  T_end = 10.0        # Near 0 K to freeze conformation
  schedule = exponential
  anneal_steps = 1.0
}
```

**Expected result**: Molecule explores rotational conformers at high T, settles into most stable conformation by end.

### Use Case 2: Protein Structure Refinement

**Goal**: Refine homology model of DHFR protein

```fnl
nsteps = 200000
dt[fs] = 0.5

thermostat ANNEAL
temperature = 300.
gamma[THz] = 10.

annealing{
  T_start = 400.0     # Conservative to preserve folding
  T_end = 300.0       # Physiological temperature
  schedule = cosine   # Gentle cooling
  anneal_steps = 0.75 # 75% anneal, 25% equilibrate
}
```

**Expected result**: Side chains relax, hydrogen bonds optimize, overall energy decreases while maintaining secondary structure.

### Use Case 3: Transition State Search

**Goal**: Find saddle point for SN2 reaction

```fnl
nsteps = 100000
dt[fs] = 0.2  # Smaller timestep for stability

thermostat ANNEAL
temperature = 300.
gamma[THz] = 5.   # Lower friction for better sampling

annealing{
  T_start = 600.0
  T_end = 200.0
  schedule = linear
  anneal_steps = 0.9
}
```

**Note**: May need constraints on reaction coordinate to stay near transition state region.

## Monitoring Annealing Progress

### 1. Energy vs Time

Plot total energy throughout simulation:
```bash
grep "Energy" md.log | awk '{print $2, $4}' > energy.dat
gnuplot -e "plot 'energy.dat' with lines"
```

**Expected**: Energy should decrease monotonically and plateau at final temperature.

### 2. Temperature vs Time

Temperature is printed in output:
```bash
grep "Temperature" md.log > temperature.dat
```

**Expected**: Should follow chosen schedule (exponential, linear, or cosine decay).

### 3. RMSD vs Time (for proteins)

Track structural changes:
```python
from fennol.utils.analysis import compute_rmsd

rmsd = compute_rmsd(trajectory, reference_structure)
```

**Expected**: Large RMSD initially (high T exploration), decreasing as system cools.

## Troubleshooting

### Problem: System explodes at high temperature

**Cause**: Timestep too large for high kinetic energies

**Solution**: Reduce timestep
```fnl
dt[fs] = 0.2  # Instead of 0.5
```

### Problem: Energy doesn't decrease

**Cause**: Cooling too fast, system trapped in local minimum

**Solution**: Slower annealing
```fnl
nsteps = 200000        # Double the steps
# OR
anneal_steps = 0.5     # Anneal over only 50%, equilibrate rest
```

### Problem: Protein denatures during annealing

**Cause**: Starting temperature too high

**Solution**: Lower T_start and use cosine schedule
```fnl
annealing{
  T_start = 350.0      # More conservative
  T_end = 300.0
  schedule = cosine    # Gentler cooling
}
```

### Problem: Final structure not stable

**Cause**: Not enough equilibration at final temperature

**Solution**: Add equilibration phase
```fnl
annealing{
  anneal_steps = 0.75  # Anneal 75%, equilibrate 25%
}
```

## Advanced Topics

### Custom Temperature Schedule (Python API)

For custom schedules, you can modify the schedule function directly:

```python
def custom_schedule(step):
    # Example: Two-stage cooling
    if step < 25000:
        # Fast initial cooling
        return 1000.0 * (0.5 ** (step / 5000))
    else:
        # Slow final cooling
        progress = (step - 25000) / 75000
        return 100.0 * (1.0 - progress) + 10.0 * progress
```

### Parallel Tempering vs Simulated Annealing

**Simulated Annealing**: Single trajectory, temperature decreases over time
- Pros: Simple, guaranteed to cool to final state
- Cons: Can get trapped in local minima

**Parallel Tempering**: Multiple replicas at different temperatures, exchanges allowed
- Pros: Better sampling, can escape local minima
- Cons: More computationally expensive, requires multiple replicas

FeNNol currently supports simulated annealing. Parallel tempering may be added in future releases.

## References

1. Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). "Optimization by simulated annealing." *Science*, 220(4598), 671-680.

2. Brünger, A. T., Adams, P. D., & Rice, L. M. (1997). "New applications of simulated annealing in X-ray crystallography and solution NMR." *Structure*, 5(3), 325-336.

3. Lee, J., Scheraga, H. A., & Rackovsky, S. (1997). "New optimization method for conformational energy calculations on polypeptides: Conformational space annealing." *Journal of Computational Chemistry*, 18(9), 1222-1232.

## Example Workflows

See example input files in `examples/md/`:

- `anneal_exponential.fnl`: Basic exponential decay example
- `anneal_linear.fnl`: Linear cooling example
- `anneal_cosine.fnl`: Cosine decay for proteins
- `anneal_protein_refine.fnl`: Complete protein refinement protocol
- `anneal_conformer_search.fnl`: Small molecule conformer search

## Getting Help

If you encounter issues:

1. Check parameter values (T_start > T_end, anneal_steps in (0,1])
2. Verify temperature schedule is appropriate for your system
3. Monitor energy and temperature during simulation
4. Try slower cooling or different schedule
5. Report bugs at: https://github.com/anthropics/FeNNol/issues

---

**Version**: 1.0
**Last updated**: November 2025
