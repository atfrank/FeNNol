# PMF File Output Guide

## Fixed Issues

✅ **PMF initialization messages now appear**  
✅ **PMF progress tracking working**  
✅ **PMF output files are automatically created**  

## PMF Output Files

Your PMF simulation will now create the following files:

### 1. Final PMF Results: `pmf_output.dat`

**When created:** Automatically at the end of simulation  
**Contents:** Final PMF curve with calculated free energy values  

**Example:**
```
# PMF Estimation Results
# Temperature: 300.0 K
#
# Restraint: backside_attack_restraint
# Collective variable: distance
# Window centers: [2.5, 2.4, 2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
# Samples per window: [210, 210, 210, 210, 210, 210, 210, 210, 210, 210]
#
# Window_Center(Å)  PMF(kcal/mol)
       2.500000        0.000000
       2.400000        0.143000
       2.300000        0.287000
       2.100000        0.574000
       2.000000        0.718000
       1.900000        0.861000
       1.800000        1.005000
       1.700000        1.148000
       1.600000        1.292000
       1.500000        1.435000
```

### 2. Real-time Data: `pmf_[restraint_name]_realtime.dat` (Optional)

**When created:** Only if you add `write_realtime_pmf = yes` to your restraint  
**Contents:** Live data collection as simulation runs  

**Example:**
```
# PMF Real-time Data for restraint: backside_attack_restraint
# Collective variable: distance
# Number of windows: 10
# Window values: [2.5, 2.4, 2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
# Columns: Step Window_Index Distance_Value
#    Step Win     Distance
      90   0     2.497245
      91   0     2.501832
      92   0     2.498773
      ...
     390   1     2.403127
     391   1     2.397854
     ...
```

## How to Enable Real-time Output

Add this line to your restraint configuration:

```
restraints {
  backside_attack_restraint {
    # ... your existing configuration ...
    write_realtime_pmf = yes    # Add this line
  }
}
```

## What You'll See During Simulation

Your simulation will now show:

### At Startup:
```
# PMF CONFIGURATION for restraint 'backside_attack_restraint':
#   Type: backside_attack
#   Interpolation mode: discrete
#   Estimate PMF: True
#   Update steps: 300
#   Equilibration: 90 steps (30%)
#   Sampling: 210 steps (70%)
#   Scanning: target_distance
#   Windows: 10
#   Distance values: [2.5, 2.4, 2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
```

### During Simulation:
```
# PMF INITIALIZATION
# Restraint: backside_attack_restraint
# Collective variable: distance
# Number of windows: 10
# Window values: [2.5, 2.4, 2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6, 1.5]
# PMF estimation: ACTIVE

# PMF [backside_attack_restraint] === STARTING SAMPLING for window 1 ===
# PMF [backside_attack_restraint] Window 1: SAMPLING 25.0% | Samples=52 | Current=2.498 | Mean=2.501±0.003

# PMF [backside_attack_restraint] === WINDOW 1 COMPLETE ===
#   Target: 2.500
#   Samples collected: 210
#   Mean ± StdDev: 2.501 ± 0.003
#   Range: [2.495, 2.507]

# PMF [backside_attack_restraint] === ENTERING WINDOW 2/10 ===
#   Target distance: 2.400
#   Equilibration: 90 steps, Sampling: 210 steps
```

### At End of Simulation:
```
# Writing PMF estimation results...
# PMF data written to pmf_output.dat

# FINAL PMF RESULTS
# Temperature: 300.0 K
# Restraint: backside_attack_restraint
# Collective variable: distance
# Total samples: 2100
# PMF barrier: 1.44 kcal/mol at window 10 (target=1.500)
```

## File Locations

Files are created in your simulation directory:
- `pmf_output.dat` - Final PMF results
- `pmf_backside_attack_restraint_realtime.dat` - Real-time data (if enabled)

## Summary

The PMF system is now fully functional:
- ✅ Messages appear during simulation
- ✅ Data is collected automatically  
- ✅ Files are created at simulation end
- ✅ Real-time output optional but available
- ✅ Final PMF curve calculated and saved

No additional setup required - just run your simulation as normal!