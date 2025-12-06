# MD Simulation Status

## Session Summary (2025-11-19)

### Problem Identified

The initial MD simulation (`run_dhfr_md_implicit.py`) was running with **ONLY implicit solvent forces** (GB/OBC model), missing the critical ANI-2x neural network force field. This caused the protein structure to completely disintegrate:

- **RMSD: 115,311 Å** (structure exploded)
- Only electrostatic + solvation forces
- No bonded interactions (bonds, angles, dihedrals)
- No van der Waals forces

### Solution Implemented

Created corrected simulation script: `run_dhfr_md_ani2x_implicit.py`

**Key Changes:**
1. Uses FeNNol's `load_model()` interface to properly load ANI-2x + GB together
2. Combines ANI-2x neural network forces with GB implicit solvent
3. Proper unit conversions (Hartree → kcal/mol, Bohr → Angstrom)
4. Handles batched energy output from FeNNol model

**Architecture:**
```
Total Force = ANI-2x Force + GB Implicit Solvent Force
              ↑                ↑
              Neural network   CUDA-optimized Phase 3C
              (bonded + vdW)   (electrostatics + solvation)
```

### Phase 3C Achievement

**CUDA Optimization Complete:**
- ✅ Phase 3A: GPU reduction kernel
- ✅ Phase 3B: dE/dR neighbor list optimization
- ✅ Phase 3C: Fused GB+dE/dR kernel
- **Total speedup: 2.48× over baseline** (30.52 ms → 12.31 ms per force evaluation)

**Key Implementation:**
- Fused kernel combining GB pairwise forces and dE/dR computation
- Single neighbor list traversal
- Reuses computed values (r, f_GB, derivatives)
- Mixed precision (FP32 intermediates, FP64 accumulators)
- Validated: max relative error 1.4e-7

### Current Simulation Status

**Script:** `run_dhfr_md_ani2x_implicit.py`

**Running:** ✅ Active (PID: 1981518)
- CPU: 406% (JIT compilation in progress)
- Memory: 2.7 GB
- Runtime: ~13 minutes (still compiling)

**Parameters:**
- System: DHFR (2,499 atoms)
- Force field: ANI-2x neural network
- Solvation: OBC Generalized Born (CUDA Phase 3C)
- Time step: 1.0 fs
- Total steps: 10,000
- Temperature: 300 K (NVE ensemble)

**Expected Behavior:**
- JIT compilation may take 15-20 minutes for first evaluation
- After compilation, ~45-60 ms/step expected
- Total simulation time: ~10-15 minutes after compilation

### Files Modified

1. **`run_dhfr_md_ani2x_implicit.py`** (new)
   - Corrected MD simulation with ANI-2x + GB
   - Uses FeNNol MD interface
   - Proper unit handling

2. **`run_dhfr_md_implicit.py`** (original - INCORRECT)
   - Only GB forces (structure explodes)
   - Keep for comparison/debugging

### Validation Results

**Phase 3C Fused Kernel:**
- Energy agreement: < 1e-6 relative error
- Force agreement: < 1e-6 relative error
- dE/dR agreement: < 1e-6 relative error
- Validated on 100-atom and 500-atom systems

**Previous MD Run (GB only):**
- Completed 10,000 steps
- Performance: 44.98 ms/step
- Energy: -18,100 kcal/mol (stable)
- Temperature: 302.65 K (perfect conservation)
- **RMSD: 115,311 Å** ⚠️ STRUCTURE EXPLODED (expected without bonded forces)

### Next Steps

1. Wait for current ANI-2x+GB simulation to complete
2. Analyze results:
   - Verify RMSD stays reasonable (<5 Å for 10 ps)
   - Check energy conservation
   - Validate structural stability
3. Compare performance: ANI-2x+GB vs GB-only
4. Document final performance metrics

### Technical Notes

**Unit Conversions:**
- Energy: Hartree → kcal/mol (× 627.5095)
- Force: Hartree/Bohr → kcal/mol/Å (× 627.5095 / 0.529177)
- Length: Bohr → Angstrom (× 0.529177)

**FeNNol Model Output:**
- Returns batched energy: shape (1,) not scalar
- Must use `jnp.sum()` or index [0] before conversion

**Tinker XYZ Format:**
- Has index column: `indexed=True`
- No comment line in header: `has_comment_line=False`
- Format: `index element x y z atom_type connectivity...`
