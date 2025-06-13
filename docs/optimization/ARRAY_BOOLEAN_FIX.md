# JAX Array Boolean Fix Documentation

## Issue
When using spherical boundary restraints with PDB-based atom selection, the following error occurred:

```
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

This error happened in the `setup_restraints` function when trying to evaluate a JAX array in a boolean context.

## Root Cause
The error occurred at this line in `src/fennol/md/restraints.py`:
```python
selected_indices = parse_atom_selection(atom_selection, pdb_data, 
                                       initial_coordinates or pdb_data.get('coordinates'))
```

JAX arrays cannot be evaluated in boolean contexts using `or` when they contain multiple elements. The expression `initial_coordinates or pdb_data.get('coordinates')` failed because `initial_coordinates` is a JAX array.

## Solution
Replaced the problematic boolean evaluation with an explicit None check:

### Before (problematic):
```python
selected_indices = parse_atom_selection(atom_selection, pdb_data, 
                                       initial_coordinates or pdb_data.get('coordinates'))
```

### After (fixed):
```python
coords_for_selection = initial_coordinates if initial_coordinates is not None else pdb_data.get('coordinates')
selected_indices = parse_atom_selection(atom_selection, pdb_data, coords_for_selection)
```

## Additional Improvements
1. **Enhanced `calculate_center_of_mass` function** to handle both NumPy and JAX arrays
2. **Proper array conversion** when calculating center of mass
3. **Consistent JAX array handling** throughout the spherical boundary restraint code

## Files Modified
- `src/fennol/md/restraints.py`: Fixed boolean evaluation and array handling
- `src/fennol/utils/io.py`: Enhanced `calculate_center_of_mass` for JAX array compatibility

## Testing
The fix was verified by running simulations with spherical boundary restraints using auto-center calculation. The restraint now works correctly:

```
# Restraint 'sphere_boundary': Auto-calculated center at [3.0457838  0.38623306 0.31810233]
# Initialized 1 restraints
```

## Impact
- ✅ Spherical boundary restraints with auto-center calculation now work correctly
- ✅ PDB-based atom selection is functional (when PDB files are provided)
- ✅ No regression in existing functionality
- ✅ Better error handling for array operations