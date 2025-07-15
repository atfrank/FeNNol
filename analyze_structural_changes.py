#!/usr/bin/env python3
"""
Analyze structural changes during TS optimization to confirm optimization is working
"""

import numpy as np
import os
import re

def parse_xyz_file(filename):
    """Parse XYZ file and return list of frames"""
    frames = []
    if not os.path.exists(filename):
        return frames
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.isdigit():
            n_atoms = int(line)
            i += 1
            comment = lines[i].strip()
            i += 1
            
            # Parse atoms
            atoms = []
            for j in range(n_atoms):
                parts = lines[i + j].split()
                symbol = parts[0]
                coords = [float(x) for x in parts[1:4]]
                atoms.append((symbol, coords))
            
            frames.append({
                'n_atoms': n_atoms,
                'comment': comment,
                'atoms': atoms
            })
            i += n_atoms
        else:
            i += 1
    
    return frames

def calculate_rmsd(coords1, coords2):
    """Calculate RMSD between two coordinate sets"""
    coords1 = np.array(coords1)
    coords2 = np.array(coords2)
    diff = coords1 - coords2
    return np.sqrt(np.mean(np.sum(diff**2, axis=1)))

def calculate_bond_length(coord1, coord2):
    """Calculate distance between two points"""
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def analyze_multimodel_trajectory(filename):
    """Analyze multi-model XYZ trajectory for structural changes"""
    print(f"\nAnalyzing trajectory: {filename}")
    print("=" * 60)
    
    frames = parse_xyz_file(filename)
    if not frames:
        print(f"No frames found in {filename}")
        return
    
    print(f"Total frames: {len(frames)}")
    
    # Extract energies and forces from comments
    energies = []
    max_forces = []
    rms_forces = []
    
    for i, frame in enumerate(frames):
        comment = frame['comment']
        
        # Parse energy
        energy_match = re.search(r'energy=([+-]?\d*\.?\d+)', comment)
        if energy_match:
            energies.append(float(energy_match.group(1)))
        
        # Parse max force
        max_force_match = re.search(r'max_force=([+-]?\d*\.?\d+)', comment)
        if max_force_match:
            max_forces.append(float(max_force_match.group(1)))
        
        # Parse RMS force
        rms_force_match = re.search(r'rms_force=([+-]?\d*\.?\d+)', comment)
        if rms_force_match:
            rms_forces.append(float(rms_force_match.group(1)))
    
    # Analyze energy progression
    if energies:
        print(f"\nEnergy progression (Ha):")
        for i, energy in enumerate(energies):
            print(f"  Frame {i}: {energy:.6f}")
        
        energy_change = energies[-1] - energies[0]
        print(f"  Total energy change: {energy_change:.6f} Ha")
    
    # Analyze force progression
    if max_forces:
        print(f"\nMax force progression:")
        for i, force in enumerate(max_forces):
            print(f"  Frame {i}: {force:.6f}")
        
        force_change = max_forces[-1] - max_forces[0]
        print(f"  Max force change: {force_change:.6f}")
    
    # Calculate structural changes (RMSD between consecutive frames)
    print(f"\nStructural changes (RMSD between consecutive frames):")
    prev_coords = None
    total_rmsd = 0
    
    for i, frame in enumerate(frames):
        coords = [atom[1] for atom in frame['atoms']]
        
        if prev_coords is not None:
            rmsd = calculate_rmsd(coords, prev_coords)
            total_rmsd += rmsd
            print(f"  Frame {i-1} -> {i}: RMSD = {rmsd:.6f} Å")
        
        prev_coords = coords
    
    if len(frames) > 1:
        avg_rmsd = total_rmsd / (len(frames) - 1)
        print(f"  Average RMSD per step: {avg_rmsd:.6f} Å")
    
    # Calculate total structural change (first to last frame)
    if len(frames) > 1:
        first_coords = [atom[1] for atom in frames[0]['atoms']]
        last_coords = [atom[1] for atom in frames[-1]['atoms']]
        total_structure_change = calculate_rmsd(first_coords, last_coords)
        print(f"  Total structural change: {total_structure_change:.6f} Å")
    
    # Analyze specific bond changes for water dimer
    if len(frames) > 1 and len(frames[0]['atoms']) == 6:
        print(f"\nWater dimer bond analysis:")
        
        # O-O distance changes
        for i, frame in enumerate(frames):
            coords = [atom[1] for atom in frame['atoms']]
            o1_coord = coords[0]  # First oxygen
            o2_coord = coords[3]  # Second oxygen
            oo_distance = calculate_bond_length(o1_coord, o2_coord)
            print(f"  Frame {i}: O-O distance = {oo_distance:.6f} Å")
        
        # H-bond distance changes (O1-H2...O2)
        print(f"\nHydrogen bond analysis:")
        for i, frame in enumerate(frames):
            coords = [atom[1] for atom in frame['atoms']]
            o1_coord = coords[0]  # First oxygen
            h2_coord = coords[2]  # H on first water pointing toward second water
            o2_coord = coords[3]  # Second oxygen
            
            # Distance from H to second oxygen
            h_bond_distance = calculate_bond_length(h2_coord, o2_coord)
            print(f"  Frame {i}: H-bond distance = {h_bond_distance:.6f} Å")

def check_optimization_errors(filename):
    """Check for optimization errors in log output"""
    print(f"\nChecking for optimization errors...")
    print("=" * 40)
    
    # Check if optimization converged properly
    if os.path.exists(filename):
        frames = parse_xyz_file(filename)
        if len(frames) > 1:
            # Check if last few frames are identical (stuck)
            if len(frames) >= 3:
                last_coords = [atom[1] for atom in frames[-1]['atoms']]
                second_last_coords = [atom[1] for atom in frames[-2]['atoms']]
                third_last_coords = [atom[1] for atom in frames[-3]['atoms']]
                
                rmsd_last = calculate_rmsd(last_coords, second_last_coords)
                rmsd_second = calculate_rmsd(second_last_coords, third_last_coords)
                
                if rmsd_last < 1e-8 and rmsd_second < 1e-8:
                    print("⚠️  WARNING: Optimization appears to be stuck (no structural changes)")
                else:
                    print("✓ Optimization shows structural changes")
            
            # Check force convergence
            frames = parse_xyz_file(filename)
            if frames:
                last_comment = frames[-1]['comment']
                max_force_match = re.search(r'max_force=([+-]?\d*\.?\d+)', last_comment)
                if max_force_match:
                    final_max_force = float(max_force_match.group(1))
                    if final_max_force > 0.1:
                        print(f"⚠️  WARNING: High final max force: {final_max_force:.6f}")
                    else:
                        print(f"✓ Reasonable final max force: {final_max_force:.6f}")
        else:
            print("⚠️  WARNING: Only one frame found - optimization may have failed")
    else:
        print(f"❌ File {filename} not found")

def main():
    """Main analysis function"""
    print("TRANSITION STATE OPTIMIZATION ANALYSIS")
    print("=" * 60)
    
    # Analyze trajectory files
    trajectory_files = [
        'water_dimer.multimodel.xyz',
        'h2_dissociation.multimodel.xyz',
        'sn2_reaction.multimodel.xyz'
    ]
    
    for trajectory_file in trajectory_files:
        if os.path.exists(trajectory_file):
            analyze_multimodel_trajectory(trajectory_file)
            check_optimization_errors(trajectory_file)
        else:
            print(f"\nFile {trajectory_file} not found - skipping analysis")
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")

if __name__ == "__main__":
    main()