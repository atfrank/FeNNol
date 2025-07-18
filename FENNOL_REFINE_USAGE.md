# FeNNol Structure Refinement Tool (fennol_refine)

The `fennol_refine` tool is a command-line interface for refining molecular structures using FeNNol neural network potentials with enhanced sampling methods. It is designed to relieve atomic clashes and optimize molecular interactions while maintaining structural integrity through RMSD restraints.

## Features

- **Enhanced Sampling Methods**: Simulated annealing, Monte Carlo, and gradient descent
- **RMSD Restraints**: Maintain structural similarity to initial coordinates
- **Clash Detection and Removal**: Automatic detection and removal of atomic clashes
- **Flexible Configuration**: Extensive command-line options for customization
- **Multiple Output Formats**: PDB and XYZ trajectory files
- **GPU Support**: CUDA acceleration for large systems

## Installation

The tool is included with FeNNol. Install FeNNol in development mode to get access to all command-line tools:

```bash
# Install FeNNol in development mode
pip install -e .

# Test installation
fennol_refine --help
```

After installation, `fennol_refine` will be available in your environment's bin directory and can be called from anywhere.

## Basic Usage

### Simple Structure Refinement

```bash
./fennol_refine --pdb input_structure.pdb --model your_model.pkl
```

### With Custom Parameters

```bash
./fennol_refine \
  --pdb input_structure.pdb \
  --model your_model.pkl \
  --method simulated_annealing \
  --initial-temperature 600 \
  --final-temperature 10 \
  --temperature-steps 100 \
  --rmsd-force-constant 10.0 \
  --clash-cutoff 2.0 \
  --output-prefix refined_structure
```

### Gradient Descent with Strong RMSD Restraint

```bash
./fennol_refine \
  --pdb input_structure.pdb \
  --model your_model.pkl \
  --method gradient_descent \
  --max-iterations 1000 \
  --force-tolerance 1e-5 \
  --rmsd-force-constant 20.0 \
  --rmsd-atoms heavy \
  --output-prefix minimized
```

### Monte Carlo Sampling

```bash
./fennol_refine \
  --pdb input_structure.pdb \
  --model your_model.pkl \
  --method monte_carlo \
  --initial-temperature 400 \
  --final-temperature 50 \
  --mc-displacement 0.1 \
  --mc-acceptance-ratio 0.5 \
  --write-trajectory \
  --trajectory-format pdb
```

## Command-Line Options

### Required Arguments

- `--pdb PDB`: Input PDB structure file
- `--model MODEL`: FeNNol model file path

### Refinement Method

- `--method {simulated_annealing,monte_carlo,gradient_descent}`: Refinement method (default: simulated_annealing)

### Temperature Schedule (for annealing methods)

- `--temperature-schedule {linear,exponential,cosine}`: Temperature schedule (default: exponential)
- `--initial-temperature TEMP`: Initial temperature in K (default: 600)
- `--final-temperature TEMP`: Final temperature in K (default: 10)
- `--temperature-steps STEPS`: Number of temperature steps (default: 100)

### Simulated Annealing Parameters

- `--annealing-cycles CYCLES`: Number of cycles per temperature (default: 10)
- `--annealing-hold-time TIME`: Steps to hold at each temperature (default: 10)

### Monte Carlo Parameters

- `--mc-displacement DISP`: Initial MC displacement in Angstroms (default: 0.1)
- `--mc-acceptance-ratio RATIO`: Target MC acceptance ratio (default: 0.5)

### RMSD Restraint Options

- `--rmsd-restraint` / `--no-rmsd-restraint`: Enable/disable RMSD restraint (default: enabled)
- `--rmsd-force-constant FC`: RMSD restraint force constant (default: 10.0)
- `--rmsd-target TARGET`: Target RMSD from initial structure (default: 0.0)
- `--rmsd-atoms {all,backbone,heavy}`: Atoms to include in RMSD calculation (default: heavy)

### Clash Detection and Removal

- `--clash-cutoff CUTOFF`: Distance cutoff for clash detection in Angstroms (default: 2.0)
- `--clash-force-constant FC`: Force constant for clash removal (default: 100.0)
- `--clash-iterations ITER`: Maximum iterations for clash removal (default: 100)

### Covalent Bond Preservation

- `--preserve-covalent-bonds` / `--no-preserve-covalent-bonds`: Enable/disable covalent bond preservation (default: enabled)
- `--bond-force-constant FC`: Force constant for bond length restraints (default: 1000.0)
- `--angle-force-constant FC`: Force constant for bond angle restraints (default: 100.0)
- `--bond-types-to-preserve TYPES`: Types of bonds to preserve - choices: backbone, sidechain, inter_residue, all (default: backbone inter_residue)
- `--max-bond-deviation DEV`: Maximum allowed bond length deviation in Angstroms (default: 0.5)
- `--detect-topology` / `--no-detect-topology`: Enable/disable automatic topology detection (default: enabled)

### Optimization Parameters

- `--max-iterations ITER`: Maximum optimization iterations (default: 1000)
- `--force-tolerance TOL`: Force convergence tolerance (default: 1e-4)
- `--energy-tolerance TOL`: Energy convergence tolerance (default: 1e-6)
- `--displacement-tolerance TOL`: Displacement convergence tolerance (default: 1e-3)
- `--max-step STEP`: Maximum step size in Angstroms (default: 0.2)

### Device and Precision

- `--device {cpu,gpu,cuda}`: Device to run on
- `--double-precision`: Use double precision (float64)
- `--matmul-precision {highest,high,float32}`: Matrix multiplication precision

### Output Options

- `--output-prefix PREFIX`: Output file prefix (default: input filename)
- `--write-trajectory` / `--no-trajectory`: Write trajectory files (both PDB and XYZ formats) (default: enabled)
- `--print-freq FREQ`: Print frequency for optimization steps (default: 10)
- `--verbose, -v`: Enable verbose output

## Output Files

The tool generates several output files:

1. **`{prefix}_refined.pdb`**: Final refined structure in PDB format
2. **`{prefix}_refined.xyz`**: Final refined structure in XYZ format
3. **`{prefix}_traj.pdb`**: Trajectory in PDB format (if `--write-trajectory` is enabled)
4. **`{prefix}_traj.xyz`**: Trajectory in XYZ format (if `--write-trajectory` is enabled)
5. **`{prefix}_multimodel.pdb`**: Multi-model PDB trajectory file (if `--write-trajectory` is enabled)
6. **`{prefix}_multimodel.xyz`**: Multi-model XYZ trajectory file (if `--write-trajectory` is enabled)

## Examples

### Example 1: Basic RNA Structure Refinement

```bash
./fennol_refine \
  --pdb examples/refine/covalent_example.pdb \
  --model models/rna_model.pkl \
  --method simulated_annealing \
  --initial-temperature 500 \
  --final-temperature 50 \
  --rmsd-atoms backbone \
  --output-prefix rna_refined \
  --verbose
```

### Example 2: Protein Side Chain Optimization

```bash
./fennol_refine \
  --pdb protein_structure.pdb \
  --model protein_model.pkl \
  --method gradient_descent \
  --max-iterations 2000 \
  --force-tolerance 1e-5 \
  --rmsd-atoms backbone \
  --rmsd-force-constant 50.0 \
  --clash-cutoff 1.8 \
  --output-prefix protein_optimized
```

### Example 3: Small Molecule Conformational Search

```bash
./fennol_refine \
  --pdb small_molecule.pdb \
  --model small_mol_model.pkl \
  --method monte_carlo \
  --initial-temperature 800 \
  --final-temperature 100 \
  --temperature-steps 200 \
  --mc-displacement 0.2 \
  --no-rmsd-restraint \
  --output-prefix conformer_search
```

### Example 4: Structure Refinement with Strict Covalent Preservation

```bash
./fennol_refine \
  --pdb protein_with_clashes.pdb \
  --model protein_model.pkl \
  --method simulated_annealing \
  --initial-temperature 400 \
  --final-temperature 25 \
  --preserve-covalent-bonds \
  --bond-force-constant 2000.0 \
  --angle-force-constant 200.0 \
  --bond-types-to-preserve backbone inter_residue sidechain \
  --max-bond-deviation 0.3 \
  --clash-cutoff 1.8 \
  --output-prefix covalent_preserved
```

## Tips and Best Practices

1. **Start with Simulated Annealing**: For most cases, simulated annealing provides a good balance between exploration and optimization.

2. **Use RMSD Restraints Carefully**: Strong RMSD restraints preserve structure but may prevent necessary conformational changes. Adjust the force constant based on your needs.

3. **Monitor Clash Removal**: Check the final output for remaining clashes. If many clashes remain, consider reducing the clash cutoff or increasing the force constant.

4. **Temperature Schedule**: 
   - Use higher initial temperatures for more exploration
   - Use exponential schedule for gradual cooling
   - Use cosine schedule for smooth transitions

5. **Atom Selection for RMSD**:
   - Use "backbone" for proteins to preserve secondary structure
   - Use "heavy" for general molecular refinement
   - Use "all" for very gentle refinement

6. **GPU Usage**: For large systems (>1000 atoms), use `--device gpu` for better performance.

7. **Trajectory Analysis**: Enable trajectory writing to analyze the refinement process and identify potential issues.

8. **Covalent Bond Preservation**:
   - Always keep `--preserve-covalent-bonds` enabled for biomolecular systems
   - Use higher bond force constants (1000-5000) for critical bonds
   - Monitor covalent integrity in the output - broken bonds indicate problems
   - For systems with disulfide bonds, include "sidechain" in `--bond-types-to-preserve`
   - Adjust `--max-bond-deviation` based on system flexibility needs

9. **Topology Detection**:
   - Automatic topology detection works well for standard biomolecules
   - For unusual molecules, consider manual topology specification
   - Check topology summary in verbose output to ensure correct detection
   - Inter-residue bonds are automatically detected for peptide and nucleic acid systems

## Troubleshooting

### Common Issues

1. **"Model file not found"**: Ensure the model file path is correct and the file exists.
2. **"PDB file not found"**: Check the PDB file path and format.
3. **Memory issues**: Use `--device cpu` for large systems or reduce the number of temperature steps.
4. **Slow performance**: Use GPU acceleration with `--device gpu` for large systems.

### Performance Tips

- Use `--double-precision` only when necessary (increases memory usage)
- Reduce `--temperature-steps` for faster refinement
- Use `--print-freq` to control output frequency
- Disable trajectory writing (`--no-trajectory`) for faster execution

## Integration with FeNNol

The `fennol_refine` tool is designed to work seamlessly with other FeNNol tools:

1. Use `fennol_ts` to find transition states, then refine with `fennol_refine`
2. Use FeNNol MD simulations to generate starting conformations
3. Combine with FeNNol analysis tools for comprehensive structure characterization

For more information about FeNNol and its capabilities, see the main FeNNol documentation.