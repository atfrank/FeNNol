#!/usr/bin/env python3
"""
Analyze transition state results for chemical correctness
"""

import numpy as np
from pathlib import Path

def read_xyz(filename):
    """Read XYZ file and return atoms and coordinates"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    natoms = int(lines[0].strip())
    comment = lines[1].strip()
    
    atoms = []
    coords = []
    
    for i in range(2, 2 + natoms):
        parts = lines[i].strip().split()
        atoms.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    
    return atoms, np.array(coords), comment

def distance(r1, r2):
    """Calculate distance between two points"""
    return np.linalg.norm(r1 - r2)

def angle(r1, r2, r3):
    """Calculate angle r1-r2-r3 in degrees"""
    v1 = r1 - r2
    v2 = r3 - r2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    # Handle numerical issues
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))

def analyze_water_dimer(atoms, coords):
    """Analyze water dimer structure"""
    print("Water Dimer Analysis:")
    print("=" * 40)
    
    # Identify atoms
    o_indices = [i for i, atom in enumerate(atoms) if atom == 'O']
    h_indices = [i for i, atom in enumerate(atoms) if atom == 'H']
    
    if len(o_indices) != 2 or len(h_indices) != 4:
        print(f"Unexpected atom count: {len(o_indices)} O, {len(h_indices)} H")
        return
    
    o1, o2 = o_indices
    
    # Find which H atoms belong to which O
    o1_h = []
    o2_h = []
    
    for h in h_indices:
        d1 = distance(coords[h], coords[o1])
        d2 = distance(coords[h], coords[o2])
        if d1 < d2:
            o1_h.append(h)
        else:
            o2_h.append(h)
    
    print(f"O1 index: {o1}, H atoms: {o1_h}")
    print(f"O2 index: {o2}, H atoms: {o2_h}")
    
    # Analyze O-H bonds
    for i, o in enumerate([o1, o2]):
        h_atoms = o1_h if i == 0 else o2_h
        for h in h_atoms:
            bond_length = distance(coords[h], coords[o])
            print(f"O{i+1}-H{h+1} bond length: {bond_length:.3f} Å")
    
    # Analyze O-O distance
    oo_distance = distance(coords[o1], coords[o2])
    print(f"O-O distance: {oo_distance:.3f} Å")
    
    # Look for hydrogen bonds
    min_hb_distance = float('inf')
    hb_donor = None
    hb_acceptor = None
    
    for h in o1_h:
        d = distance(coords[h], coords[o2])
        if d < min_hb_distance:
            min_hb_distance = d
            hb_donor = f"O{1}-H{h+1}"
            hb_acceptor = f"O{2}"
    
    for h in o2_h:
        d = distance(coords[h], coords[o1])
        if d < min_hb_distance:
            min_hb_distance = d
            hb_donor = f"O{2}-H{h+1}"
            hb_acceptor = f"O{1}"
    
    print(f"Shortest H-bond: {hb_donor} ... {hb_acceptor} = {min_hb_distance:.3f} Å")
    
    # Check if this looks like a reasonable water dimer TS
    if 2.5 < oo_distance < 3.5:
        print("✓ O-O distance reasonable for water dimer")
    else:
        print("✗ O-O distance unusual for water dimer")
    
    if 1.5 < min_hb_distance < 2.5:
        print("✓ Hydrogen bond distance reasonable")
    else:
        print("✗ Hydrogen bond distance unusual")
    
    return True

def analyze_h2_dissociation(atoms, coords):
    """Analyze H2 dissociation structure"""
    print("\nH2 Dissociation Analysis:")
    print("=" * 40)
    
    if len(atoms) != 2 or atoms[0] != 'H' or atoms[1] != 'H':
        print(f"Unexpected atoms: {atoms}")
        return
    
    # H-H distance
    hh_distance = distance(coords[0], coords[1])
    print(f"H-H distance: {hh_distance:.3f} Å")
    
    # Check if this looks like a reasonable H2 TS
    if 1.0 < hh_distance < 2.0:
        print("✓ H-H distance reasonable for H2 dissociation TS")
    else:
        print("✗ H-H distance unusual for H2 dissociation TS")
    
    # Expected H2 bond length is ~0.74 Å, TS should be stretched
    if hh_distance > 0.9:
        print("✓ H-H bond is stretched (expected for TS)")
    else:
        print("✗ H-H bond not stretched enough for TS")
    
    return True

def analyze_sn2_reaction(atoms, coords):
    """Analyze SN2 reaction structure"""
    print("\nSN2 Reaction Analysis:")
    print("=" * 40)
    
    # Expected: F, C, H, H, H, Cl
    expected_atoms = ['F', 'C', 'H', 'H', 'H', 'Cl']
    if len(atoms) != 6:
        print(f"Unexpected atom count: {len(atoms)}")
        return
    
    # Find key atoms
    f_idx = None
    c_idx = None
    cl_idx = None
    h_indices = []
    
    for i, atom in enumerate(atoms):
        if atom == 'F':
            f_idx = i
        elif atom == 'C':
            c_idx = i
        elif atom == 'Cl':
            cl_idx = i
        elif atom == 'H':
            h_indices.append(i)
    
    if f_idx is None or c_idx is None or cl_idx is None:
        print(f"Missing key atoms: F={f_idx}, C={c_idx}, Cl={cl_idx}")
        return
    
    print(f"F index: {f_idx}, C index: {c_idx}, Cl index: {cl_idx}")
    print(f"H indices: {h_indices}")
    
    # Key distances
    fc_distance = distance(coords[f_idx], coords[c_idx])
    c_cl_distance = distance(coords[c_idx], coords[cl_idx])
    
    print(f"F-C distance: {fc_distance:.3f} Å")
    print(f"C-Cl distance: {c_cl_distance:.3f} Å")
    
    # SN2 reaction coordinate
    reaction_coord = fc_distance - c_cl_distance
    print(f"Reaction coordinate (F-C) - (C-Cl): {reaction_coord:.3f} Å")
    
    # Check C-H bonds
    for h in h_indices:
        ch_distance = distance(coords[c_idx], coords[h])
        print(f"C-H{h+1} bond length: {ch_distance:.3f} Å")
    
    # Check F-C-Cl angle
    fcl_angle = angle(coords[f_idx], coords[c_idx], coords[cl_idx])
    print(f"F-C-Cl angle: {fcl_angle:.1f}°")
    
    # Check if this looks like a reasonable SN2 TS
    if 1.8 < fc_distance < 2.8 and 1.8 < c_cl_distance < 2.8:
        print("✓ F-C and C-Cl distances reasonable for SN2 TS")
    else:
        print("✗ F-C or C-Cl distances unusual for SN2 TS")
    
    if abs(reaction_coord) < 0.5:
        print("✓ Reaction coordinate close to zero (expected for TS)")
    else:
        print("✗ Reaction coordinate far from zero")
    
    if 160 < fcl_angle < 180:
        print("✓ F-C-Cl angle close to linear (expected for SN2 TS)")
    else:
        print("✗ F-C-Cl angle not linear enough for SN2 TS")
    
    return True

def main():
    """Main analysis function"""
    print("TRANSITION STATE STRUCTURE ANALYSIS")
    print("=" * 60)
    
    # Check for result files
    result_files = {
        'water_dimer.ts.xyz': analyze_water_dimer,
        'h2_dissociation.ts.xyz': analyze_h2_dissociation,
        'sn2_reaction.ts.xyz': analyze_sn2_reaction
    }
    
    for filename, analyzer in result_files.items():
        if Path(filename).exists():
            try:
                atoms, coords, comment = read_xyz(filename)
                print(f"\nFile: {filename}")
                print(f"Comment: {comment}")
                analyzer(atoms, coords)
            except Exception as e:
                print(f"Error analyzing {filename}: {e}")
        else:
            print(f"\nFile {filename} not found - skipping analysis")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()