# Spherical Boundary Restraint Guide

## Overview

The spherical boundary restraint is designed to maintain atoms within (or outside) a spherical region. This is particularly useful for:

- Simulating spherical droplets without periodic boundary conditions
- Maintaining the shape of nanoparticles or clusters
- Creating spherical cavities in systems
- Controlling the expansion/compression of molecular systems
- Protein simulations with advanced PDB-based atom selection
- Maintaining specific regions of biomolecules

## Basic Usage

### Keep Atoms Inside a Sphere

```
restraints {
  sphere_boundary {
    type = spherical_boundary
    center = 0.0, 0.0, 0.0  # Center coordinates [x, y, z] in Angstroms
    radius = 15.0           # Sphere radius in Angstroms
    force_constant = 10.0   # Force constant (higher = stronger restraint)
    atoms = all             # Apply to all atoms (or specify list: [0, 1, 2, ...])
    mode = outside          # "outside" keeps atoms inside sphere
    style = harmonic        # Restraint style
  }
}
```

### Keep Atoms Outside a Sphere (Create Cavity)

```
restraints {
  sphere_cavity {
    type = spherical_boundary
    center = 5.0, 5.0, 5.0  # Cavity center
    radius = 3.0            # Cavity radius
    force_constant = 50.0   # Strong force to maintain cavity
    atoms = all
    mode = inside           # "inside" keeps atoms outside sphere
    style = harmonic
  }
}
```

## Parameters

### Required Parameters

- `type`: Must be `spherical_boundary`
- `radius`: Radius of the sphere in Angstroms

### Optional Parameters

- `center`: Center of the sphere
  - `[x, y, z]`: Explicit coordinates in Angstroms
  - `auto`: Auto-calculate from selected atoms (default if not specified)
- `force_constant`: Strength of the restraint (default: 10.0)
- `atoms`: Which atoms to apply restraint to
  - `all`: Apply to all atoms (default)
  - List of indices: `[0, 1, 2, 3, ...]`
  - Dictionary with selection criteria (see PDB-based selection)
- `mode`: Restraint mode
  - `outside`: Keep atoms inside the sphere (default)
  - `inside`: Keep atoms outside the sphere
- `style`: Type of restraint potential
  - `harmonic`: Standard harmonic restraint (default)
  - `flat_bottom`: No force within tolerance of radius
  - `one_sided`: Force only in one direction
- `tolerance`: For flat_bottom style, width of flat region (default: 0.0)
- `side`: For one_sided style, `upper` or `lower` (default: `lower`)

### PDB-Based Selection Parameters

- `pdb_file`: Path to PDB file for atom information
- `atom_selection`: Dictionary with selection criteria
  - `residue_numbers`: List of residue numbers or ranges
  - `residue_names`: List of residue names
  - `atom_names`: List of atom names
  - `chain_ids`: List of chain identifiers
  - `within_distance`: Distance-based selection
    - `distance`: Cutoff distance in Angstroms
    - `of`: Selection criteria for reference atoms

## Advanced Features

### Time-Varying Radius

You can change the sphere radius during simulation:

```
restraints {
  expanding_sphere {
    type = spherical_boundary
    center = 0.0, 0.0, 0.0
    radius = 10.0, 12.0, 14.0, 16.0, 18.0, 20.0  # Expand from 10 to 20 Å
    force_constant = 5.0
    atoms = all
    update_steps = 1000                          # Steps between radius changes
    interpolation_mode = interpolate             # Smooth transition
  }
}
```

### Time-Varying Force Constant

You can also vary the restraint strength:

```
restraints {
  adaptive_boundary {
    type = spherical_boundary
    center = 0.0, 0.0, 0.0
    radius = 15.0
    force_constant = 1.0, 5.0, 10.0, 20.0  # Gradually strengthen restraint
    atoms = all
    update_steps = 2500
  }
}
```

### Flat-Bottom Restraint

Allow free movement within a tolerance of the sphere radius:

```
restraints {
  soft_boundary {
    type = spherical_boundary
    center = 0.0, 0.0, 0.0
    radius = 20.0
    force_constant = 10.0
    atoms = all
    style = flat_bottom
    tolerance = 2.0  # No force between r=18-22 Å
  }
}
```

## PDB-Based Atom Selection

The spherical boundary restraint supports advanced atom selection using PDB file information:

### Basic PDB Selection

```
restraints {
  protein_core {
    type = spherical_boundary
    pdb_file = protein.pdb
    center = auto  # Auto-calculate from selected atoms
    
    atom_selection {
      # Select specific residues
      residue_numbers = 10, 20, 30, "40-50"  # Individual or ranges
      
      # Select by residue name
      residue_names = ALA, GLY, VAL
      
      # Select by atom name
      atom_names = CA, CB, CG  # Alpha carbon, beta carbon, etc.
      
      # Select by chain
      chain_ids = A, B
    }
    
    radius = 25.0
    force_constant = 5.0
  }
}
```

### Distance-Based Selection

Select atoms within a certain distance of reference atoms:

```
restraints {
  active_site_region {
    type = spherical_boundary
    pdb_file = enzyme.pdb
    center = auto
    
    atom_selection {
      within_distance {
        distance = 12.0  # 12 Angstrom cutoff
        of {
          residue_numbers = 157  # Active site residue
          atom_names = CA
        }
      }
    }
    
    radius = 18.0
    force_constant = 3.0
  }
}
```

### Combined Selection Criteria

All criteria are combined with AND logic:

```
atom_selection {
  residue_numbers = "50-100"  # Residues 50-100
  atom_names = CA, C, N, O    # AND backbone atoms only
  chain_ids = A               # AND in chain A
}
```

### Auto-Center Calculation

When `center = auto`, the center is calculated as the center of mass of selected atoms:
- Uses actual atomic masses if available
- Falls back to geometric center if masses not available
- Calculated only once at the beginning of simulation

## Collective Variables

Monitor distances from the sphere center:

```
colvars {
  atom0_from_center {
    type = sphere_distance
    atom_indices = 0
    center = 0.0, 0.0, 0.0
  }
  
  # Monitor multiple atoms
  surface_atoms {
    type = sphere_distance
    atom_indices = 0, 10, 20, 30, 40
    center = 0.0, 0.0, 0.0
  }
}
```

## Use Cases

### 1. Water Droplet Simulation

Maintain a spherical water droplet shape:

```
restraints {
  droplet {
    type = spherical_boundary
    center = 0.0, 0.0, 0.0
    radius = 25.0
    force_constant = 5.0
    atoms = all
    style = flat_bottom
    tolerance = 1.0
  }
}
```

### 2. Nanoparticle Compression

Compress a nanoparticle by reducing sphere radius:

```
restraints {
  compress_particle {
    type = spherical_boundary
    center = 0.0, 0.0, 0.0
    radius = 30.0, 25.0, 20.0, 15.0
    force_constant = 10.0
    atoms = all
    update_steps = 5000
    interpolation_mode = interpolate
  }
}
```

### 3. Maintain Protein Shape

Keep a protein compact during simulation:

```
restraints {
  protein_shape {
    type = spherical_boundary
    center = 15.0, 15.0, 15.0  # Protein center of mass
    radius = 35.0
    force_constant = 2.0
    atoms = all
    style = flat_bottom
    tolerance = 5.0  # Allow 5 Å fluctuation
  }
}
```

### 4. Create Solvent Cavity

Maintain a cavity in solvent for ligand binding studies:

```
restraints {
  binding_cavity {
    type = spherical_boundary
    center = 10.0, 10.0, 10.0  # Binding site center
    radius = 5.0
    force_constant = 100.0     # Strong force to maintain cavity
    atoms = all
    mode = inside              # Keep atoms outside
  }
}
```

## Tips and Best Practices

1. **Choose appropriate force constants**: Start with small values (1-10) and increase if needed
2. **Use flat_bottom for soft restraints**: This prevents artificial forces on atoms near the boundary
3. **Monitor the restraint energy**: Large restraint energies indicate the system is fighting the constraint
4. **Center placement**: Choose the center based on your system's center of mass or geometric center
5. **Gradual changes**: When using time-varying parameters, ensure smooth transitions to avoid system perturbation

## Troubleshooting

- **System explodes**: Reduce force_constant or use flat_bottom style
- **Atoms escape boundary**: Increase force_constant or reduce tolerance
- **Unnatural dynamics**: Use flat_bottom with appropriate tolerance
- **Performance issues**: Consider applying restraint only to surface atoms instead of all atoms