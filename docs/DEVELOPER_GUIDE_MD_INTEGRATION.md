# FeNNol MD Integration - Developer Guide

**Version:** 1.0
**Last Updated:** November 18, 2025
**Author:** Development Team

This guide documents the MD (Molecular Dynamics) integration architecture, recent improvements, and best practices for using the FeNNol MD simulation framework.

---

## Table of Contents

1. [MD Integration Architecture](#md-integration-architecture)
2. [Trajectory File Handling](#trajectory-file-handling)
3. [Energy Reporting](#energy-reporting)
4. [Using fennol_md Command](#using-fennol_md-command)
5. [Parameter Files (.fnl)](#parameter-files-fnl)
6. [Common Issues and Fixes](#common-issues-and-fixes)
7. [Best Practices](#best-practices)

---

## 1. MD Integration Architecture

### Overview

The FeNNol MD simulation framework consists of two main modules that work together to provide a complete molecular dynamics simulation capability:

- **`integrate.py`**: Handles initialization of dynamics, integrators, thermostats, and force calculations
- **`dynamic.py`**: Manages the main simulation loop, I/O, trajectory output, and energy reporting

### File Locations

```
src/fennol/md/
├── integrate.py       # Integrator initialization and force computation
├── dynamic.py         # Main simulation loop and I/O
├── initial.py         # System initialization
├── thermostats.py     # Temperature control
├── barostats.py       # Pressure control
├── restraints.py      # Geometric restraints
└── colvars.py         # Collective variables
```

### How integrate.py Works

**Purpose**: Initialize the MD integrator, setup force calculations, and configure physical models.

**Key Functions**:

1. **`initialize_dynamics()` (line 53-68)**
   - Entry point called by `dynamic.py`
   - Initializes integrator and system state
   - Returns: `step`, `update_conformation`, `dyn_state`, `system`

2. **`initialize_integrator()` (line 71-933)**
   - Sets up timestep, masses, and dynamics parameters
   - Configures thermostat (NVE, Langevin, QTB, etc.)
   - Configures barostat if using PBC
   - Initializes neighbor lists
   - Sets up collective variables and restraints
   - **Initializes implicit solvent (GB) if configured**
   - Returns integrator step functions

3. **`initialize_implicit_solvent()` (line 37-50)**
   - Creates GB/OBC implicit solvent model
   - Called automatically when `implicit_solvent` parameters present
   - Returns GB model object or None

**Implicit Solvent Integration** (lines 214-250):

When implicit solvent is enabled, the integrator:
- Assigns atomic charges (user-provided or defaults)
- Adds charges to the conformation dictionary
- Enables GB force calculations in `update_forces()`

**Force Calculation** (lines 539-694 for classical MD):

The `update_forces()` function:
1. Computes ML potential energy and forces
2. **Adds GB solvation energy and forces** (lines 565-635)
   - Converts units: kcal/mol → Hartree (factor: 0.001593601)
   - Stores `gb_energy` in system dict for reporting
3. Adds restraint forces if configured
4. Updates collective variables

### How dynamic.py Works

**Purpose**: Run the main MD simulation loop, handle I/O, and report energies.

**Key Functions**:

1. **`main()` (line 49-96)**
   - Parses command line arguments
   - Loads parameter file (.fnl)
   - Sets device (CPU/GPU) and precision
   - Calls `dynamic()` to run simulation

2. **`dynamic()` (line 99-740)**
   - Loads model and system data
   - Calls `initialize_dynamics()` from `integrate.py`
   - Runs main simulation loop
   - Handles trajectory and energy output

**Main Simulation Loop** (lines 363-725):

```python
for istep in range(1, nsteps + 1):
    # Update system (calls integrate.py step function)
    dyn_state, system, conformation, preproc_state, model_output = step(...)

    # Print energies every nprint steps
    if istep % nprint == 0:
        # Extract and print energies (including GB energy)

    # Write trajectory frame every ndump steps
    if istep % ndump == 0:
        # Write coordinates to trajectory file

    # Print summary statistics every nsummary steps
    if istep % nsummary == 0:
        # Print averages and performance metrics
```

### Integration Between Modules

```
fennol_md command (entry point)
    ↓
dynamic.main()
    ↓ parses .fnl file
dynamic.dynamic()
    ↓ initializes system
integrate.initialize_dynamics()
    ↓ sets up integrator
integrate.initialize_integrator()
    ↓ creates step function
    ├── initialize_implicit_solvent() (if configured)
    ├── setup_restraints() (if configured)
    └── setup_colvars() (if configured)
    ↓ returns step function
dynamic.dynamic() simulation loop
    ↓ calls step() each iteration
    ├── update_forces() (from integrate.py)
    │   ├── ML potential forces
    │   ├── GB forces (if enabled)
    │   └── Restraint forces (if enabled)
    ├── stepA() - half velocity update, position update
    ├── stepB() - half velocity update, compute kinetic energy
    └── Output energies and trajectory
```

---

## 2. Trajectory File Handling

### The Problem (FIXED)

**Issue**: Trajectory files were being opened in append mode, causing data from multiple runs to accumulate in the same file. This led to:
- Confusing trajectory data when running multiple simulations
- Difficulty analyzing individual runs
- Unexpected file sizes

### The Fix

**Location**: `src/fennol/md/dynamic.py`, lines 319-347

**Implementation**:

```python
# Delete existing trajectory files to avoid appending to old runs
import os
if write_all_beads:
    for i in range(nbeads):
        traj_file = f"{system_name}_bead{i+1:03d}" + traj_ext
        if os.path.exists(traj_file):
            os.remove(traj_file)
    fout = [
        open(f"{system_name}_bead{i+1:03d}" + traj_ext, "w") for i in range(nbeads)
    ]
else:
    traj_file = system_name + traj_ext
    if os.path.exists(traj_file):
        os.remove(traj_file)
    fout = open(system_name + traj_ext, "w")
```

**What This Does**:
1. Checks if trajectory file already exists
2. Deletes old file if present
3. Opens new file in write mode ('w')
4. Ensures each simulation run starts with a fresh trajectory file

**Also Applied To**:
- Ensemble weights file (line 337-340)
- Centroid trajectory file (line 344-347)
- Pickle data file (line 262-265)

### Supported Trajectory Formats

The system supports three trajectory formats:

1. **ARC** (Tinker Archive) - Default
   - File extension: `.arc`
   - Includes coordinates, cell parameters, and properties

2. **XYZ** (Simple XYZ)
   - File extension: `.traj.xyz`
   - Basic coordinate format

3. **ExtXYZ** (Extended XYZ)
   - File extension: `.traj.extxyz`
   - Includes additional properties in comment line

Configure in .fnl file:
```
traj_format arc  # or xyz, extxyz
```

---

## 3. Energy Reporting

### GB Energy Display Integration

**Problem Solved**: GB (Generalized Born) implicit solvent energy was being calculated but not displayed in MD output, making it difficult to monitor solvation effects during simulation.

### Implementation Details

#### 3.1 Header Modification

**Location**: `src/fennol/md/dynamic.py`, lines 286-289

```python
# Check if GB implicit solvent is enabled
has_gb_model = "implicit_solvent" in simulation_parameters
if has_gb_model:
    header += "         EGB"
```

**What This Does**:
- Checks if `implicit_solvent` section exists in parameter file
- Adds "EGB" column to output header if GB is enabled
- Automatically adjusts header width for proper alignment

**Example Output Header**:
```
#     Step   Time[ps]        Etot        Epot        Ekin    Temp[K]         EGB
```

#### 3.2 Energy Tracking in properties_traj

**Location**: `src/fennol/md/dynamic.py`, lines 392-414

```python
# Track GB solvation energy if present
has_gb = "gb_energy" in system
if has_gb:
    gb_energy = system["gb_energy"]

properties_traj[f"Etot[{atom_energy_unit_str}]"].append(
    etot * atom_energy_unit
)
properties_traj[f"Epot[{atom_energy_unit_str}]"].append(
    epot * atom_energy_unit
)
properties_traj[f"Ekin[{atom_energy_unit_str}]"].append(
    ek * atom_energy_unit
)
# ... other energies ...
if has_gb:
    properties_traj[f"EGB[{atom_energy_unit_str}]"].append(
        gb_energy * atom_energy_unit
    )
```

**What This Does**:
1. Checks if `gb_energy` key exists in system dictionary
2. Extracts GB energy value (in kcal/mol)
3. Converts to display units (respects `per_atom_energy` setting)
4. Stores in properties trajectory for averaging

**Note**: The GB energy is stored by `integrate.py` during force calculation:
- Location: `src/fennol/md/integrate.py`, line 629
- `new_sys["gb_energy"] = gb_energy`

#### 3.3 Print Line Formatting

**Location**: `src/fennol/md/dynamic.py`, lines 430-432

```python
if has_gb:
    line += f"  {gb_energy*atom_energy_unit: #10.4f}"
```

**Format Specification**:
- `#10.4f`: Fixed-point format with 10 characters width, 4 decimal places
- Always includes sign and decimal point (# flag)
- Right-aligned with padding
- Consistent with other energy columns

**Example Output**:
```
     10000      5.000    -234.5678   -240.1234      5.5556     300.00    -12.3456
```
Where `-12.3456` is the GB energy in the configured units.

### Unit Conversion

**Important**: GB energy undergoes multiple unit conversions:

1. **CUDA Kernel Output**: kcal/mol
   - Location: `src/fennol/cuda/src/gb_energy_forces.cu`

2. **Stored in System**: kcal/mol (total, not per-atom)
   - Location: `src/fennol/md/integrate.py`, line 629
   - `new_sys["gb_energy"] = gb_energy`

3. **Display Conversion**:
   - Multiplied by `atom_energy_unit` for display
   - If `per_atom_energy = yes`: divided by number of atoms
   - If `energy_unit = kcal/mol`: no further conversion needed
   - If `energy_unit = hartree`: multiply by 0.001593601

Example calculation:
```python
# For a 100-atom system with per_atom_energy = yes
gb_energy_total = -123.456  # kcal/mol from CUDA
atom_energy_unit = 1.0 / 100  # per-atom factor
displayed_value = gb_energy_total * atom_energy_unit  # -1.23456 kcal/mol/atom
```

### Summary Statistics

GB energy is included in summary output (every `nsummary` steps):

```
# Averages over last 10000 steps :
#   Etot       :  -234.57    +/-  12.345  kcal/mol/atom
#   Epot       :  -240.12    +/-  10.234  kcal/mol/atom
#   Ekin       :    5.556    +/-   2.111  kcal/mol/atom
#   EGB        :  -12.346    +/-   1.234  kcal/mol/atom
```

This provides statistics on GB energy fluctuations during the simulation.

---

## 4. Using fennol_md Command

### Command Line Interface

The `fennol_md` command is the official entry point for MD simulations.

**Installation**:
- Defined in `pyproject.toml`, line 47:
  ```toml
  fennol_md = "fennol.md.dynamic:main"
  ```
- Installed automatically with `pip install -e .`

**Usage**:
```bash
fennol_md parameter_file.fnl
```

**Example**:
```bash
# Run MD simulation with implicit solvent
fennol_md examples/md/dhfr/input_implicit_solvent.fnl

# Run on GPU
fennol_md examples/md/watersmall/input.fnl
```

### Why Use fennol_md?

**Benefits**:
1. **Proper initialization**: Handles device setup, precision, and configuration
2. **Consistent I/O**: Ensures trajectory files are handled correctly
3. **Error handling**: Provides clear error messages
4. **Maintained interface**: Updates automatically with package improvements

**Avoid Custom Scripts Unless**:
- You need specialized initialization
- You're debugging integrator internals
- You're implementing new features

### Entry Point Flow

```
$ fennol_md input.fnl
    ↓
pyproject.toml entry point
    ↓
dynamic.main()
    ↓ parse arguments (line 55-58)
parse_input(args.param_file)
    ↓ load .fnl file
set device and precision (lines 61-79)
    ↓
dynamic.dynamic(simulation_parameters, device, fprec)
    ↓
[Main simulation runs]
```

### Alternative: Direct Python Usage

If you need programmatic access:

```python
from fennol.md.dynamic import dynamic
from fennol.utils.input_parser import parse_input

# Load parameters
params = parse_input("input.fnl")

# Run simulation
dynamic(params, device="cpu", fprec="float32")
```

**Note**: You must manually set JAX device and precision configuration.

---

## 5. Parameter Files (.fnl)

### File Format

FeNNol uses a custom `.fnl` (Fennol) format for parameter files. The format supports:
- Key-value pairs: `key = value`
- Nested sections: `section_name{ ... }`
- Units: `key[unit] = value`
- Comments: `#` or `//`

### Basic Structure

```fnl
# Device configuration
device cuda:0
matmul_prec highest

# Model
model_file path/to/model.fnx

# Input coordinates
xyz_input{
  file structure.xyz
  indexed yes
  has_comment_line no
}

# Simulation parameters
nsteps = 10000
dt[fs] = 0.5

# Thermostat
thermostat LGV
temperature = 300.0
gamma[THz] = 10.0
```

### Implicit Solvent Configuration

**Location**: Any `.fnl` file
**Section name**: `implicit_solvent`

**Full Example**:

```fnl
# Implicit solvent (OBC Generalized Born)
implicit_solvent{
  model OBC                    # Model type: OBC, GBn, or HCT
  dielectric 80.0              # Solvent dielectric constant (water = 80.0)
  cutoff 12.0                  # Cutoff distance in Angstroms
  surface_tension 0.005        # Surface tension coefficient
  probe_radius 1.4             # Solvent probe radius (water = 1.4 Å)
  radii_set mbondi             # Atomic radii set: mbondi, mbondi2, bondi
  include_nonpolar yes         # Include nonpolar (SASA) term
}
```

**Parameters Explained**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | string | OBC | GB model: OBC (default), GBn, HCT |
| `dielectric` | float | 80.0 | Solvent dielectric constant (water = 78.5-80.0) |
| `cutoff` | float | 12.0 | Cutoff distance for GB calculations (Angstroms) |
| `surface_tension` | float | 0.005 | SASA surface tension (kcal/mol/Å²) |
| `probe_radius` | float | 1.4 | Solvent probe radius (Angstroms, water = 1.4) |
| `radii_set` | string | mbondi | Atomic radii: mbondi (recommended), mbondi2, bondi |
| `include_nonpolar` | bool | yes | Include nonpolar (SASA) solvation term |

**Periodic Boundary Conditions**:

When using implicit solvent, you typically want:
```fnl
minimum_image no         # Disable PBC
wrap_box no              # Don't wrap coordinates
estimate_pressure no     # Can't compute pressure without PBC
```

**Example Files**:

1. **DHFR Protein** (`examples/md/dhfr/input_implicit_solvent.fnl`):
   - Large biomolecule (2489 atoms)
   - No PBC, implicit solvent only
   - 12 Å cutoff for efficiency

2. **Single Water Molecule** (`water_ani_gb.fnl`):
   - Small molecule test
   - 8 Å cutoff (shorter for speed)
   - Nonpolar term disabled

### Complete Example: Protein MD with Implicit Solvent

```fnl
# Hardware
device cuda:0
matmul_prec highest

# Model
model_file ../ani2x.fnx

# Structure
xyz_input{
  file protein.xyz
  indexed yes
  has_comment_line no
}

# No periodic boundaries for implicit solvent
minimum_image no
wrap_box no
estimate_pressure no

# Generalized Born implicit solvent
implicit_solvent{
  model OBC
  dielectric 80.0
  cutoff 12.0
  surface_tension 0.005
  probe_radius 1.4
  radii_set mbondi
  include_nonpolar yes
}

# Simulation length: 1 ns
nsteps = 2000000
dt[fs] = 0.5

# Neighbor list optimization
nblist_skin 2.0

# Output: save every 1 ps
traj_format arc
tdump[ps] = 1.0

# Print every 100 steps (50 fs)
nprint = 100
nsummary = 10000

# Langevin thermostat (NVT ensemble)
thermostat LGV
temperature = 300.0
gamma[THz] = 10.0

# Energy units
energy_unit kcal/mol
per_atom_energy yes
```

---

## 6. Common Issues and Fixes

### Issue 1: Trajectory Files Appending Instead of Overwriting

**Status**: ✅ FIXED

**Symptoms**:
- Trajectory files grow unexpectedly large
- Multiple simulation runs appear in same file
- Difficult to analyze individual runs

**Root Cause**:
Trajectory files were opened in append mode without checking for existing files.

**Fix Applied**:
- Location: `src/fennol/md/dynamic.py`, lines 319-347
- Now deletes existing trajectory files before opening new ones
- Applied to: `.arc`, `.traj.xyz`, `.traj.extxyz`, ensemble weights, centroid files

**Code**:
```python
# Delete existing trajectory files to avoid appending to old runs
import os
traj_file = system_name + traj_ext
if os.path.exists(traj_file):
    os.remove(traj_file)
fout = open(system_name + traj_ext, "w")
```

**Workaround** (if using older version):
```bash
# Manually delete old trajectory files before running
rm -f system_name.arc system_name.traj.xyz
fennol_md input.fnl
```

---

### Issue 2: GB Energy Not Showing in Output

**Status**: ✅ FIXED

**Symptoms**:
- GB implicit solvent enabled but energy not displayed
- Only Etot, Epot, Ekin shown in output
- Difficult to monitor solvation effects

**Root Cause**:
- GB energy was calculated and stored in `system["gb_energy"]`
- But not added to output header or print lines

**Fix Applied**:
- Location: `src/fennol/md/dynamic.py`
- Lines 286-289: Add "EGB" to header
- Lines 392-414: Track GB energy in properties
- Lines 430-432: Add GB energy to output line

**Code**:
```python
# Header modification
has_gb_model = "implicit_solvent" in simulation_parameters
if has_gb_model:
    header += "         EGB"

# Energy tracking
has_gb = "gb_energy" in system
if has_gb:
    gb_energy = system["gb_energy"]
    properties_traj[f"EGB[{atom_energy_unit_str}]"].append(
        gb_energy * atom_energy_unit
    )

# Print line
if has_gb:
    line += f"  {gb_energy*atom_energy_unit: #10.4f}"
```

**Verification**:
Run simulation with implicit solvent and check for "EGB" column:
```
#     Step   Time[ps]        Etot        Epot        Ekin    Temp[K]         EGB
     100      0.050    -234.5678   -240.1234      5.5556     300.00    -12.3456
```

---

### Issue 3: CUDA Device Detection When No GPU Present

**Status**: ✅ FIXED

**Symptoms**:
- Error when trying to use CUDA on CPU-only system
- JAX fails to find GPU device
- Simulation crashes at startup

**Root Cause**:
- Device detection didn't properly fall back to CPU
- CUDA environment variables not properly set

**Fix Applied**:
- Location: `src/fennol/md/dynamic.py`, lines 61-74
- Properly handle CPU vs GPU device selection
- Set `CUDA_VISIBLE_DEVICES=""` for CPU mode

**Code**:
```python
device: str = simulation_parameters.get("device", "cpu").lower()
if device == "cpu":
    device = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
elif device.startswith("cuda") or device.startswith("gpu"):
    if ":" in device:
        num = device.split(":")[-1]
        os.environ["CUDA_VISIBLE_DEVICES"] = num
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = "gpu"

_device = jax.devices(device)[0]
jax.config.update("jax_default_device", _device)
```

**Parameter File Configuration**:
```fnl
# For CPU-only systems
device cpu

# For GPU systems
device cuda:0    # Use GPU 0
device cuda:1    # Use GPU 1
device gpu       # Use default GPU
```

**Verification**:
```python
import jax
print(jax.devices())  # Should show CPU or GPU as configured
```

---

### Issue 4: Unit Conversion Errors in GB Energy

**Status**: ✅ FIXED (November 18, 2025)

**Symptoms**:
- GB energy values too large or too small
- Energy drift in simulations with implicit solvent
- Incorrect solvation free energies

**Root Cause**:
- Inconsistent unit handling between CUDA (kcal/mol) and JAX (Hartree)
- Missing conversion factor: 1 kcal/mol = 0.001593601 Hartree

**Fix Applied**:
- Location: `src/fennol/md/integrate.py`, lines 386-426, 611-634
- Convert GB energy from kcal/mol to Hartree
- Apply same conversion to forces (kcal/mol/Å → Hartree/Bohr)

**Code**:
```python
# Convert GB energy and forces from kcal/mol to Hartree
kcal_to_hartree = 0.001593601

gb_energy_au = gb_energy * kcal_to_hartree
gb_forces_au = gb_forces * kcal_to_hartree

# Add GB energy to total potential energy (per-atom)
new_sys["epot"] = new_sys["epot"] + gb_energy_au / natoms

# Add GB forces to total forces
new_sys["forces"] = new_sys["forces"] + gb_forces_au

# Store GB energy for reporting (keep in kcal/mol)
new_sys["gb_energy"] = gb_energy
```

**Important Note**:
- Internal calculations use Hartree (atomic units)
- Display uses kcal/mol or other configured units
- GB energy stored in `system["gb_energy"]` is in kcal/mol (total, not per-atom)

---

### Issue 5: NaN in GB Forces (JAX Autodiff Issue)

**Status**: ⚠️ KNOWN LIMITATION

**Symptoms**:
- GB forces contain NaN values
- Simulation becomes unstable
- Positions or velocities explode

**Root Cause**:
- JAX autodiff can produce NaN for certain GB kernel implementations
- Related to chain rule derivatives in complex CUDA kernels

**Current Workaround**:
- Location: `src/fennol/md/integrate.py`, lines 592-598
- Detect NaN in forces and disable GB forces (energy still computed)

**Code**:
```python
# Check for NaN in forces (JAX autodiff issue)
import numpy as np
if np.any(np.isnan(gb_forces)):
    if not hasattr(update_forces, '_nan_warned'):
        print("WARNING: GB forces contain NaN (JAX autodiff issue). Disabling GB forces.")
        print("         GB energy will still be computed but forces set to zero.")
        update_forces._nan_warned = True
    gb_forces = jnp.zeros_like(coords)
```

**Status**:
- GB energy calculation works correctly
- GB forces disabled when NaN detected
- Simulation continues with ML potential forces only

**Future Work**:
- Investigate JAX-compatible GB force implementations
- Consider pure CUDA implementation without autodiff
- Alternative: Use finite differences for forces

---

## 7. Best Practices

### 7.1 Always Use fennol_md Interface

**Do**:
```bash
fennol_md input.fnl
```

**Don't**:
```python
# Avoid direct imports and manual setup
from fennol.md.integrate import initialize_dynamics
from fennol.md.dynamic import dynamic
# ... manual device setup ...
```

**Why**:
- Proper initialization and error handling
- Consistent file cleanup (trajectory files)
- Automatic device and precision configuration
- Future compatibility

### 7.2 Parameter File Organization

**Recommended Structure**:
```fnl
# 1. Hardware configuration (top)
device cuda:0
matmul_prec highest
double_precision  # optional

# 2. Model and input files
model_file path/to/model.fnx
xyz_input{ ... }

# 3. Periodic boundaries (if applicable)
cell = ...
minimum_image yes/no

# 4. Implicit solvent (if applicable)
implicit_solvent{ ... }

# 5. Simulation parameters
nsteps = ...
dt[fs] = ...

# 6. Neighbor lists
nblist_skin ...

# 7. Output settings
traj_format ...
tdump[ps] = ...
nprint = ...

# 8. Thermostat/Barostat
thermostat ...
temperature = ...
```

### 7.3 Implicit Solvent Configuration

**For Proteins/Large Molecules**:
```fnl
implicit_solvent{
  model OBC
  dielectric 80.0       # Water
  cutoff 12.0           # Typical for proteins
  surface_tension 0.005
  probe_radius 1.4
  radii_set mbondi      # Standard for proteins
  include_nonpolar yes  # Include SASA term
}
```

**For Small Molecules (testing)**:
```fnl
implicit_solvent{
  model OBC
  dielectric 80.0
  cutoff 8.0            # Shorter cutoff for speed
  include_nonpolar no   # Disable for simplicity
}
```

**Always disable PBC when using implicit solvent**:
```fnl
minimum_image no
wrap_box no
estimate_pressure no
```

### 7.4 Trajectory File Management

**Clean Previous Runs**:
- The system now automatically deletes old trajectory files
- No manual cleanup needed

**Multiple Runs**:
```bash
# Rename output before next run (optional)
mv system.arc system_run1.arc

# Or use different system names in .fnl files
xyz_input{
  file structure.xyz
  # Output will be named after file basename
}
```

**Archiving Results**:
```bash
# Create run-specific directory
mkdir run_20251118
mv system.arc run_20251118/
mv system.colvars run_20251118/
```

### 7.5 Monitoring Simulations

**Energy Conservation (NVE)**:
Check total energy drift:
```
# Etot should be constant (< 0.1% drift)
#     Step   Time[ps]        Etot        Epot        Ekin
     1000      0.500    -234.5678   -240.1234      5.5556
    10000      5.000    -234.5690   -240.1245      5.5555
```

**Temperature Control (NVT)**:
Check temperature stability:
```
# Temperature should fluctuate around setpoint
# Averages over last 10000 steps :
#   Temper     :     300.12    +/-   5.234  Kelvin
```

**GB Energy Tracking**:
Monitor solvation effects:
```
# EGB should be negative (favorable solvation)
# Magnitude depends on system size and charge distribution
#   EGB        :  -12.346    +/-   1.234  kcal/mol/atom
```

**Performance Metrics**:
```
# Look for consistent performance
# Perf.: 2.50 ns/day  ( 50.00 step/s )
```

### 7.6 Debugging Tips

**Enable Verbose Output**:
```fnl
nblist_verbose           # Print neighbor list info
print_timings yes        # Show timing breakdown
```

**Check Device Usage**:
```python
import jax
print(jax.devices())     # Verify correct device
```

**Monitor Memory**:
```bash
# For GPU
nvidia-smi -l 1

# For CPU
htop
```

**Validate GB Energy**:
```fnl
# Run short test with known system
nsteps = 100
nprint = 10
nsummary = 100

# Check for:
# 1. EGB column appears
# 2. Values are reasonable (negative, ~-1 to -20 kcal/mol/atom)
# 3. No NaN warnings
```

### 7.7 Production Runs

**Checklist**:
- [ ] Parameter file reviewed and validated
- [ ] Device set correctly (CPU vs GPU)
- [ ] Trajectory format chosen (arc recommended)
- [ ] Output frequency set appropriately
- [ ] Implicit solvent parameters verified
- [ ] Previous trajectory files backed up if needed
- [ ] Sufficient disk space available
- [ ] Estimated run time calculated

**Run Command**:
```bash
# Use nohup for long runs
nohup fennol_md input.fnl > output.log 2>&1 &

# Monitor progress
tail -f output.log
```

**Performance Estimation**:
```
Time per step = (system size) × (complexity factor)

For 1000 atoms with GB implicit solvent on GPU:
  ~0.02 seconds/step
  ~50 steps/second
  ~4.3 ns/day

For 3000 atoms:
  ~0.06 seconds/step
  ~16 steps/second
  ~1.4 ns/day
```

---

## Summary

This guide documents the MD integration improvements made to FeNNol, including:

1. **Architecture**: Clear separation between `integrate.py` (initialization) and `dynamic.py` (simulation loop)
2. **Trajectory Handling**: Automatic deletion of old trajectory files prevents data accumulation
3. **Energy Reporting**: GB energy now properly displayed in output with correct unit conversion
4. **Command Interface**: Use `fennol_md` command for all simulations
5. **Parameter Files**: Comprehensive `.fnl` configuration with implicit solvent support
6. **Bug Fixes**: Addressed trajectory appending, GB display, and CUDA device detection

**Key Takeaway**: Always use the `fennol_md` command with proper `.fnl` parameter files for reliable, reproducible MD simulations.

---

## Related Documentation

- **Implicit Solvent Implementation**: `docs/IMPLICIT_SOLVENT_DESIGN.md`
- **GB CUDA Optimizations**: `docs/GB_CUDA_OPTIMIZATIONS.md`
- **Restraints Guide**: `docs/restraints/RESTRAINTS_USER_GUIDE.md`
- **PMF Calculations**: `docs/pmf/PMF_USER_GUIDE.md`

---

**Questions or Issues?**

If you encounter problems not covered in this guide, please:
1. Check the issue tracker
2. Review related documentation files
3. Enable verbose output and print_timings for debugging
4. Report issues with complete .fnl file and error messages
