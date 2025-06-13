# PMF Temperature Fix Documentation

## Issue
When running FeNNol MD simulations with PMF estimation enabled, the following error occurred:
```
# Writing PMF estimation results...
# Warning: Failed to write PMF output: name 'temperature' is not defined
```

## Root Cause
In `src/fennol/md/dynamic.py`, at the end of the simulation (lines 694-712), the code attempts to write PMF output but the `temperature` variable was not properly defined. The functions `write_pmf_output()` and `print_pmf_final_summary()` in `src/fennol/md/restraints.py` expect a `temperature` parameter in Kelvin.

## Solution
The fix extracts the temperature from either:
1. The `temperature` parameter in the simulation input file (if specified)
2. Calculates it from `system_data["kT"]` using the atomic units conversion factor

### Code Changes in `src/fennol/md/dynamic.py`:

```python
# Write PMF output if PMF estimation was active
try:
    from fennol.md.restraints import pmf_data, write_pmf_output, print_pmf_final_summary
    if pmf_data:
        print("\n# Writing PMF estimation results...")
        # Extract temperature from simulation parameters or calculate from kT
        # The temperature parameter is already available in simulation_parameters
        temperature_param = simulation_parameters.get("temperature", None)
        if temperature_param is not None:
            temperature = temperature_param  # Already in Kelvin
        else:
            # Calculate from kinetic energy per atom: kT = kB * T
            # system_data["kT"] is in atomic units, convert to temperature
            # kT / kB = T, where kB in atomic units = 1/au.KELVIN
            temperature = system_data["kT"] * au.KELVIN
        write_pmf_output("pmf_output.dat", temperature=temperature)
        print_pmf_final_summary(temperature=temperature)
except Exception as e:
    print(f"# Warning: Failed to write PMF output: {e}")
```

## Affected Functions
- `write_pmf_output(filename: str = "pmf_output.dat", temperature: float = 300.0)` in `src/fennol/md/restraints.py:245`
- `print_pmf_final_summary(temperature: float = 300.0)` in `src/fennol/md/restraints.py:408`

Both functions use the temperature parameter to calculate PMF values using the relation kB*T.

## Testing
To test the fix, run a simulation with PMF estimation enabled:

```bash
python -m fennol.md.dynamic your_input.fnl
```

The input file should include restraints with PMF estimation:
```
restraints {
  backside_attack_restraint {
    type = backside_attack
    # ... other parameters ...
    estimate_pmf = yes
    equilibration_ratio = 0.3
    update_steps = 1000
    write_realtime_pmf = yes
  }
}
```

After the simulation completes, you should see:
- "# Writing PMF estimation results..." 
- "# PMF data written to pmf_output.dat"
- No temperature error message

The `pmf_output.dat` file will contain the temperature used for the calculation.