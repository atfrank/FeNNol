#!/usr/bin/env python3
"""
Analyze SN2 reaction coordinate progression during TS optimization
"""

import numpy as np
import re

def parse_multimodel_xyz(filename):
    """Parse multi-model XYZ file and extract coordinates"""
    frames = []
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

def calculate_sn2_reaction_coordinate(coords, nu_idx=0, c_idx=1, lg_idx=5):
    """Calculate SN2 reaction coordinate RC = d(Nu-C) - d(C-LG)"""
    coords = np.array(coords)
    
    nu_pos = coords[nu_idx]
    c_pos = coords[c_idx]
    lg_pos = coords[lg_idx]
    
    nu_c_dist = np.linalg.norm(nu_pos - c_pos)
    c_lg_dist = np.linalg.norm(c_pos - lg_pos)
    
    reaction_coord = nu_c_dist - c_lg_dist
    
    return reaction_coord, nu_c_dist, c_lg_dist

def calculate_angle(pos1, pos2, pos3):
    """Calculate angle between three points (pos2 is center)"""
    v1 = pos1 - pos2
    v2 = pos3 - pos2
    
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    
    return angle

def analyze_sn2_trajectory(filename):
    """Analyze SN2 trajectory for reaction coordinate progression"""
    print(f"SN2 REACTION COORDINATE ANALYSIS")
    print("=" * 60)
    
    frames = parse_multimodel_xyz(filename)
    if not frames:
        print(f"No frames found in {filename}")
        return
    
    print(f"Total frames: {len(frames)}")
    print(f"System: F⁻ + CH₃Cl → FCH₃ + Cl⁻")
    print(f"Reaction coordinate: RC = d(F-C) - d(C-Cl)")
    print()
    
    print("Frame  |  RC (Å)  | F-C (Å) | C-Cl (Å) | F-C-Cl (°) | Energy (Ha) | RMSD (Å)")
    print("-" * 80)
    
    prev_coords = None
    total_rmsd = 0
    
    for i, frame in enumerate(frames):
        coords = [atom[1] for atom in frame['atoms']]
        
        # Calculate reaction coordinate (F=0, C=1, Cl=5)
        rc, nu_c_dist, c_lg_dist = calculate_sn2_reaction_coordinate(coords, 0, 1, 5)
        
        # Calculate F-C-Cl angle
        f_pos = np.array(coords[0])
        c_pos = np.array(coords[1])
        cl_pos = np.array(coords[5])
        fcl_angle = calculate_angle(f_pos, c_pos, cl_pos)
        
        # Extract energy from comment
        energy = None
        comment = frame['comment']
        energy_match = re.search(r'energy=([+-]?\d*\.?\d+)', comment)
        if energy_match:
            energy = float(energy_match.group(1))
        
        # Calculate RMSD from previous frame
        rmsd = 0.0
        if prev_coords is not None:
            coords_array = np.array(coords)
            prev_coords_array = np.array(prev_coords)
            rmsd = np.sqrt(np.mean(np.sum((coords_array - prev_coords_array)**2, axis=1)))
            total_rmsd += rmsd
        
        print(f"{i:5d}  | {rc:8.3f} | {nu_c_dist:7.3f} | {c_lg_dist:8.3f} | {fcl_angle:10.1f} | {energy:11.6f} | {rmsd:8.4f}")
        
        prev_coords = coords
    
    if len(frames) > 1:
        avg_rmsd = total_rmsd / (len(frames) - 1)
        print("-" * 80)
        print(f"Average RMSD per step: {avg_rmsd:.4f} Å")
        
        # Calculate total structural change
        first_coords = np.array([atom[1] for atom in frames[0]['atoms']])
        last_coords = np.array([atom[1] for atom in frames[-1]['atoms']])
        total_change = np.sqrt(np.mean(np.sum((last_coords - first_coords)**2, axis=1)))
        print(f"Total structural change: {total_change:.4f} Å")
        
        # Initial vs final reaction coordinate
        initial_rc, initial_nu_c, initial_c_lg = calculate_sn2_reaction_coordinate(
            [atom[1] for atom in frames[0]['atoms']], 0, 1, 5)
        final_rc, final_nu_c, final_c_lg = calculate_sn2_reaction_coordinate(
            [atom[1] for atom in frames[-1]['atoms']], 0, 1, 5)
        
        print()
        print("REACTION COORDINATE EVOLUTION:")
        print(f"Initial RC: {initial_rc:.3f} Å (F-C: {initial_nu_c:.3f} Å, C-Cl: {initial_c_lg:.3f} Å)")
        print(f"Final RC:   {final_rc:.3f} Å (F-C: {final_nu_c:.3f} Å, C-Cl: {final_c_lg:.3f} Å)")
        print(f"RC change:  {final_rc - initial_rc:.3f} Å")
        
        # Check if approaching transition state
        if abs(final_rc) < 0.3:
            print("✓ Approaching symmetric transition state (RC ≈ 0)")
        elif final_rc > 0:
            print("→ Early transition state (F-C > C-Cl)")
        else:
            print("→ Late transition state (F-C < C-Cl)")

if __name__ == "__main__":
    analyze_sn2_trajectory("sn2_reaction.multimodel.xyz")