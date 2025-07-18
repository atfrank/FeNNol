#!/usr/bin/env python3
"""
Generate Fennol MD simulation scripts for simulated annealing with distance restraints.

This script creates restraints between:
1. Connected atoms within residues (tight restraints)
2. Atoms between different residues (loose restraints)
"""

import argparse
import numpy as np
from collections import defaultdict
import os


def parse_pdb(pdb_file):
    """Parse PDB file to extract atom information and connectivity."""
    atoms = []
    residues = defaultdict(list)
    
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_index = int(line[6:11]) - 1  # Convert to 0-indexed
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21]
                residue_num = int(line[22:26])
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                
                residue_key = (chain_id, residue_num, residue_name)
                atoms.append({
                    'index': atom_index,
                    'name': atom_name,
                    'residue': residue_key,
                    'coords': np.array([x, y, z])
                })
                residues[residue_key].append(atom_index)
    
    return atoms, residues


def get_connected_atoms_in_residue(atoms, residue_atoms):
    """Determine connected atoms within a residue based on distance cutoffs."""
    connected_pairs = []
    residue_atom_data = [atom for atom in atoms if atom['index'] in residue_atoms]
    
    # Distance cutoffs for different bond types (in Angstroms)
    bond_cutoffs = {
        ('C', 'C'): 1.6,
        ('C', 'N'): 1.5,
        ('C', 'O'): 1.5,
        ('N', 'N'): 1.5,
        ('N', 'O'): 1.5,
        ('O', 'O'): 1.5,
        ('P', 'O'): 1.7,
        ('C', 'P'): 1.9,
        ('C', 'H'): 1.2,
        ('N', 'H'): 1.1,
        ('O', 'H'): 1.1,
    }
    
    for i in range(len(residue_atom_data)):
        for j in range(i + 1, len(residue_atom_data)):
            atom1 = residue_atom_data[i]
            atom2 = residue_atom_data[j]
            
            # Get atom types (first character of atom name)
            type1 = atom1['name'][0]
            type2 = atom2['name'][0]
            
            # Find appropriate cutoff
            pair_types = [(type1, type2), (type2, type1)]
            cutoff = 1.8  # Default cutoff
            for pair_type in pair_types:
                if pair_type in bond_cutoffs:
                    cutoff = bond_cutoffs[pair_type]
                    break
            
            # Calculate distance
            dist = np.linalg.norm(atom1['coords'] - atom2['coords'])
            
            if dist < cutoff:
                connected_pairs.append((atom1['index'], atom2['index'], dist))
    
    return connected_pairs


def get_inter_residue_pairs(atoms, residues, min_distance=0.0, max_distance=5.0, allowed_atom_names=None):
    """Find atom pairs between different residues within a distance range and atom name filters."""
    inter_residue_pairs = []
    residue_keys = list(residues.keys())
    
    for i in range(len(residue_keys)):
        for j in range(i + 1, len(residue_keys)):
            res1_atoms = residues[residue_keys[i]]
            res2_atoms = residues[residue_keys[j]]
            
            for atom1_idx in res1_atoms:
                for atom2_idx in res2_atoms:
                    atom1 = next(a for a in atoms if a['index'] == atom1_idx)
                    atom2 = next(a for a in atoms if a['index'] == atom2_idx)
                    
                    # Check if atom names are allowed (if filter is specified)
                    if allowed_atom_names is not None:
                        # At least one atom must be in the allowed list
                        if atom1['name'] not in allowed_atom_names and atom2['name'] not in allowed_atom_names:
                            continue
                    
                    dist = np.linalg.norm(atom1['coords'] - atom2['coords'])
                    
                    if min_distance <= dist <= max_distance:
                        inter_residue_pairs.append((atom1_idx, atom2_idx, dist))
    
    return inter_residue_pairs


def generate_fennol_script(coord_file, output_file, temperatures, intra_force_constant, 
                          inter_force_constant, nsteps=5000, dt=0.5, tdump=0.005,
                          model_file="./mace_mp_large.fnx", traj_format="xyz",
                          thermostat="ADQTB", gamma=10.0, intra_max_distance=None,
                          inter_min_distance=0.0, inter_max_distance=5.0,
                          inter_atom_names=None, rmsd_restraint=False, rmsd_target=0.0,
                          rmsd_force_constant=0.1, rmsd_reference_file=None,
                          rmsd_residue_names=None, rmsd_style="flat_bottom", rmsd_tolerance=1.5,
                          nprint=10):
    """Generate Fennol MD simulation script with restraints."""
    
    # Parse coordinate file
    pdb_file = coord_file if coord_file.endswith('.pdb') else None
    xyz_file = coord_file if coord_file.endswith('.xyz') else None
    
    if pdb_file:
        atoms, residues = parse_pdb(pdb_file)
    else:
        raise ValueError("Currently only PDB files are supported for automatic restraint generation")
    
    # Get intra-residue connections
    intra_residue_restraints = []
    for residue_key, residue_atoms in residues.items():
        connected_pairs = get_connected_atoms_in_residue(atoms, residue_atoms)
        # Filter by max distance if specified
        if intra_max_distance is not None:
            connected_pairs = [(a1, a2, d) for a1, a2, d in connected_pairs if d <= intra_max_distance]
        intra_residue_restraints.extend(connected_pairs)
    
    # Get inter-residue pairs
    inter_residue_restraints = get_inter_residue_pairs(atoms, residues, inter_min_distance, inter_max_distance, inter_atom_names)
    
    # Write Fennol script
    with open(output_file, 'w') as f:
        # Header
        f.write("device cuda:0\n")
        f.write("enable_x64\n")
        f.write("matmul_prec highest\n")
        f.write("print_timings yes\n")
        f.write("restraint_debug true\n")
        f.write("nreplicas 1\n\n")
        
        # Model
        f.write(f"model_file {model_file}\n\n")
        
        # Input coordinates
        f.write("xyz_input{\n")
        if xyz_file:
            f.write(f"  file {xyz_file}\n")
        else:
            f.write(f"  file {os.path.splitext(pdb_file)[0]}.xyz\n")
        f.write("  indexed no\n")
        f.write("  has_comment_line yes\n")
        f.write("}\n\n")
        
        # Box settings
        f.write("minimum_image no\n")
        f.write("wrap_box no\n")
        f.write("estimate_pressure no\n\n")
        
        # MD parameters
        f.write(f"# number of steps to perform\n")
        f.write(f"nsteps = {nsteps}\n")
        f.write(f"# timestep of the dynamics\n")
        f.write(f"dt[fs] = {dt}\n")
        f.write(f"traj_format {traj_format}\n\n")
        
        f.write("nblist_skin 1.\n\n")
        
        f.write(f"#time between each saved frame\n")
        f.write(f"tdump[ps] = {tdump}\n")
        f.write(f"nprint = {nprint}\n")
        f.write("nsummary = 100\n")
        f.write("nblist_verbose\n\n")
        
        # Thermostat
        f.write(f"## set the thermostat\n")
        f.write(f"thermostat {thermostat}\n\n")
        
        f.write("## Thermostat parameters\n")
        f.write(f"temperature = {temperatures[0]}\n")
        f.write(f"#friction constant\n")
        f.write(f"gamma[THz] = {gamma}\n\n")
        
        # QTB parameters if using ADQTB
        if thermostat == "ADQTB":
            f.write("## parameters for the Quantum Thermal Bath\n")
            f.write("qtb{\n")
            f.write("  tseg[ps]=0.25\n")
            f.write("  omegacut[cm1]=15000.\n")
            f.write("  skipseg = 5\n")
            f.write("  startsave = 50\n")
            f.write("  agamma = 1.\n")
            f.write("}\n\n")
        
        # Restraints section
        f.write("# Restraints section\n")
        f.write("restraints {\n")
        
        # Intra-residue restraints (tight)
        restraint_count = 1
        for atom1, atom2, dist in intra_residue_restraints:
            f.write(f"  intra_residue_restraint{restraint_count} {{\n")
            f.write(f"    type = distance\n")
            f.write(f"    atom1 = {atom1}\n")
            f.write(f"    atom2 = {atom2}\n")
            f.write(f"    target = {dist:.3f}\n")
            f.write(f"    style = harmonic\n")
            f.write(f"    force_constant = {intra_force_constant}\n")
            f.write(f"  }}\n")
            restraint_count += 1
        
        # Inter-residue restraints (loose)
        for atom1, atom2, dist in inter_residue_restraints:
            f.write(f"  inter_residue_restraint{restraint_count} {{\n")
            f.write(f"    type = distance\n")
            f.write(f"    atom1 = {atom1}\n")
            f.write(f"    atom2 = {atom2}\n")
            f.write(f"    target = {dist:.3f}\n")
            f.write(f"    style = flat_bottom\n")
            f.write(f"    tolerance = 1.0  # Allow more movement\n")
            f.write(f"    force_constant = {inter_force_constant}\n")
            f.write(f"  }}\n")
            restraint_count += 1
        
        # RMSD restraint if enabled
        if rmsd_restraint and rmsd_reference_file:
            f.write(f"  topology_rmsd {{\n")
            f.write(f"    type = rmsd\n")
            f.write(f"    target_rmsd = {rmsd_target}\n")
            f.write(f"    force_constant = {rmsd_force_constant}\n")
            f.write(f"    reference_file = {rmsd_reference_file}\n")
            if rmsd_residue_names:
                f.write(f"    atom_selection {{\n")
                f.write(f"      residue_names = {', '.join(rmsd_residue_names)}\n")
                f.write(f"    }}\n")
            f.write(f"    style = {rmsd_style}\n")
            f.write(f"    tolerance = {rmsd_tolerance}\n")
            f.write(f"  }}\n")
        
        f.write("}\n\n")
        
        # Collective variables
        f.write("# Collective variables to track\n")
        f.write("colvars {\n")
        
        # Track a few representative restraints
        cv_count = 1
        for atom1, atom2, dist in intra_residue_restraints[:5]:  # Track first 5 intra-residue
            f.write(f"  intra_residue_distance{cv_count} {{\n")
            f.write(f"    type = distance\n")
            f.write(f"    atom1 = {atom1}\n")
            f.write(f"    atom2 = {atom2}\n")
            f.write(f"  }}\n")
            cv_count += 1
        
        for atom1, atom2, dist in inter_residue_restraints[:5]:  # Track first 5 inter-residue
            f.write(f"  inter_residue_distance{cv_count} {{\n")
            f.write(f"    type = distance\n")
            f.write(f"    atom1 = {atom1}\n")
            f.write(f"    atom2 = {atom2}\n")
            f.write(f"  }}\n")
            cv_count += 1
        
        f.write("}\n")
    
    print(f"Generated Fennol script: {output_file}")
    print(f"  - {len(intra_residue_restraints)} intra-residue restraints")
    print(f"  - {len(inter_residue_restraints)} inter-residue restraints")
    if rmsd_restraint and rmsd_reference_file:
        print(f"  - RMSD restraint enabled (target: {rmsd_target}, force_constant: {rmsd_force_constant})")
        if rmsd_residue_names:
            print(f"    - Residue selection: {', '.join(rmsd_residue_names)}")
        print(f"    - Reference file: {rmsd_reference_file}")
        print(f"    - Style: {rmsd_style} (tolerance: {rmsd_tolerance})")


def main():
    parser = argparse.ArgumentParser(description="Generate Fennol MD simulation scripts with distance restraints")
    parser.add_argument('coord_file', help='Input coordinate file (PDB or XYZ format)')
    parser.add_argument('output_file', help='Output Fennol script file (.fnl)')
    parser.add_argument('--temperatures', nargs='+', type=float, default=[300.0],
                       help='Temperature(s) for simulated annealing (default: [300.0])')
    parser.add_argument('--intra-force-constant', type=float, default=10.0,
                       help='Force constant for intra-residue restraints (default: 10.0)')
    parser.add_argument('--inter-force-constant', type=float, default=1.0,
                       help='Force constant for inter-residue restraints (default: 1.0)')
    parser.add_argument('--intra-max-distance', type=float, default=None,
                       help='Maximum distance for intra-residue restraints (default: automatic bond detection)')
    parser.add_argument('--inter-min-distance', type=float, default=0.0,
                       help='Minimum distance for inter-residue restraints (default: 0.0)')
    parser.add_argument('--inter-max-distance', type=float, default=5.0,
                       help='Maximum distance for inter-residue restraints (default: 5.0)')
    parser.add_argument('--inter-atom-names', nargs='+', type=str, default=None,
                       help='Atom names to include in inter-residue restraints (e.g., P O5\' O3\')')
    parser.add_argument('--nsteps', type=int, default=5000,
                       help='Number of MD steps (default: 5000)')
    parser.add_argument('--dt', type=float, default=0.5,
                       help='Time step in fs (default: 0.5)')
    parser.add_argument('--tdump', type=float, default=0.005,
                       help='Time between saved frames in ps (default: 0.005)')
    parser.add_argument('--model-file', default='./mace_mp_large.fnx',
                       help='Path to model file (default: ./mace_mp_large.fnx)')
    parser.add_argument('--thermostat', default='ADQTB',
                       choices=['NVE', 'NOSE', 'LGV', 'ADQTB'],
                       help='Thermostat type (default: ADQTB)')
    parser.add_argument('--gamma', type=float, default=10.0,
                       help='Friction constant in THz (default: 10.0)')
    parser.add_argument('--rmsd-restraint', action='store_true',
                       help='Enable RMSD restraint')
    parser.add_argument('--rmsd-target', type=float, default=0.0,
                       help='Target RMSD value (default: 0.0)')
    parser.add_argument('--rmsd-force-constant', type=float, default=0.1,
                       help='Force constant for RMSD restraint (default: 0.1)')
    parser.add_argument('--rmsd-reference-file', type=str, default=None,
                       help='Reference file for RMSD restraint (required if --rmsd-restraint is used)')
    parser.add_argument('--rmsd-residue-names', nargs='+', type=str, default=None,
                       help='Residue names for RMSD atom selection (e.g., C G)')
    parser.add_argument('--rmsd-style', type=str, default='flat_bottom',
                       choices=['flat_bottom', 'harmonic'],
                       help='RMSD restraint style (default: flat_bottom)')
    parser.add_argument('--rmsd-tolerance', type=float, default=1.5,
                       help='Tolerance for flat_bottom RMSD restraint (default: 1.5)')
    parser.add_argument('--nprint', type=int, default=10,
                       help='Frequency of printing simulation info (default: 10)')
    
    args = parser.parse_args()
    
    # Validate RMSD restraint arguments
    if args.rmsd_restraint and not args.rmsd_reference_file:
        parser.error("--rmsd-reference-file is required when --rmsd-restraint is enabled")
    
    generate_fennol_script(
        args.coord_file,
        args.output_file,
        args.temperatures,
        args.intra_force_constant,
        args.inter_force_constant,
        args.nsteps,
        args.dt,
        args.tdump,
        args.model_file,
        'xyz',
        args.thermostat,
        args.gamma,
        args.intra_max_distance,
        args.inter_min_distance,
        args.inter_max_distance,
        args.inter_atom_names,
        args.rmsd_restraint,
        args.rmsd_target,
        args.rmsd_force_constant,
        args.rmsd_reference_file,
        args.rmsd_residue_names,
        args.rmsd_style,
        args.rmsd_tolerance,
        args.nprint
    )


if __name__ == "__main__":
    main()