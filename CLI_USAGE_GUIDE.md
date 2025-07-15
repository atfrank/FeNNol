# fennol_ts CLI Usage Guide

The `fennol_ts` command-line interface now supports running transition state optimizations without needing to write input files. You can use it in two ways:

## 1. Traditional Mode (with input file)
```bash
fennol_ts input.fnl
```

## 2. CLI-Only Mode (no input file required)
```bash
fennol_ts --xyz structure.xyz --model model.fnx --method dimer
```

## Required Parameters for CLI-Only Mode
- `--xyz FILE`: XYZ structure file containing initial geometry
- `--model FILE`: Model file path (e.g., MACE model)
- `--method METHOD`: TS optimization method (`quasi_newton`, `dimer`, or `sn2`)

## Example Commands

### H2 Dissociation with Dimer Method
```bash
JAX_PLATFORMS=cpu fennol_ts \
    --xyz h2_molecule.xyz \
    --model /path/to/mace_model.fnx \
    --method dimer \
    --max-iterations 50 \
    --device cpu \
    --verbose
```

### Water Dimer with Quasi-Newton Method
```bash
JAX_PLATFORMS=cpu fennol_ts \
    --xyz water_dimer.xyz \
    --model /path/to/mace_model.fnx \
    --method quasi_newton \
    --ts-trust-radius 0.2 \
    --ts-max-uphill-steps 3 \
    --max-iterations 100 \
    --device cpu
```

### SN2 Reaction with SN2 Method
```bash
JAX_PLATFORMS=cpu fennol_ts \
    --xyz sn2_system.xyz \
    --model /path/to/mace_model.fnx \
    --method sn2 \
    --sn2-nu-index 1 \
    --sn2-c-index 2 \
    --sn2-lg-index 6 \
    --sn2-target-nu-c-distance 2.2 \
    --sn2-target-c-lg-distance 2.2 \
    --max-iterations 30 \
    --device cpu \
    --verbose
```

## Common Options

### General Parameters
- `--max-iterations N`: Maximum optimization iterations (default: 500)
- `--force-tolerance TOL`: Force convergence tolerance (default: 1e-4)
- `--output-prefix PREFIX`: Output file prefix (default: XYZ filename)
- `--device DEVICE`: Device to use (`cpu`, `gpu`, `cuda`)
- `--verbose`: Enable verbose output
- `--print-freq N`: Print frequency for optimization steps (default: 1)

### Precision Options
- `--double-precision`: Use double precision (float64)
- `--matmul-precision PREC`: Matrix multiplication precision (`highest`, `high`, `float32`)

### Output Options
- `--no-multimodel`: Disable multi-model XYZ trajectory writing
- `--no-pdb`: Disable PDB trajectory writing

## Method-Specific Parameters

### Quasi-Newton Method
- `--ts-trust-radius RADIUS`: Trust radius for optimization (default: 0.3)
- `--ts-max-uphill-steps N`: Maximum uphill steps allowed (default: 5)
- `--ts-initial-hessian-scale SCALE`: Initial Hessian scale factor (default: 0.1)

### Dimer Method
- `--dimer-separation DIST`: Dimer separation distance (default: 0.01)
- `--dimer-rotation-tolerance TOL`: Dimer rotation tolerance (default: 0.1)
- `--dimer-max-rotations N`: Maximum dimer rotations (default: 10)

### SN2 Method
- `--sn2-nu-index INDEX`: Nucleophile atom index (1-based, required)
- `--sn2-c-index INDEX`: Carbon atom index (1-based, required)
- `--sn2-lg-index INDEX`: Leaving group atom index (1-based, required)
- `--sn2-target-nu-c-distance DIST`: Target Nu-C distance (default: 2.2)
- `--sn2-target-c-lg-distance DIST`: Target C-LG distance (default: 2.2)

## Important Notes

1. **JAX Platform**: When using CPU, set `JAX_PLATFORMS=cpu` to avoid GPU initialization errors:
   ```bash
   JAX_PLATFORMS=cpu fennol_ts ...
   ```

2. **File Paths**: Use absolute paths for reliability, especially for the model file.

3. **SN2 Atom Indices**: For SN2 reactions, atom indices are 1-based (not 0-based).

4. **Output Files**: The tool generates multiple output files:
   - `.ts.xyz`: Final transition state structure
   - `.multimodel.xyz`: Optimization trajectory
   - `.traj.pdb`: PDB trajectory
   - `.best_ts_guess.xyz/.pdb`: Best TS guess found during optimization

5. **Convergence**: If optimization doesn't converge, try:
   - Increasing `--max-iterations`
   - Adjusting trust radius parameters
   - Using a better initial guess structure

## Override Input Files

You can also use CLI arguments to override parameters in existing input files:
```bash
fennol_ts input.fnl --method dimer --max-iterations 100 --verbose
```

This allows you to modify specific parameters without editing the input file.