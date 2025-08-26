# Multi-Site Backside Attack Restraint

## Overview

The multi-site backside attack restraint extends the standard backside restraint to handle competitive nucleophilic substitution reactions where multiple nucleophilic sites can potentially react with a single electrophile. This is particularly useful for:

- **RNA/DNA selectivity studies**: Understanding which nucleophilic sites (2'-OH, N7, N3) are preferred
- **Covalent inhibitor design**: Exploring selectivity between multiple cysteine residues
- **Mechanistic studies**: Investigating competitive SN2 reactions

## Key Features

### Soft Competition Model
- All specified nucleophilic sites receive restraint forces simultaneously
- Natural dynamics determine site preference rather than hard switching
- Smooth, continuous forces prevent discontinuities

### Intelligent Weighting Schemes
- **Distance-based**: Closer nucleophiles receive stronger restraints
- **Geometry-based**: Sites with better SN2 geometry get higher weights
- **Hybrid**: Combines distance and geometry factors

### Real-Time Selectivity Tracking
- Monitor reaction coordinates for all sites
- Track which site is "winning" the competition
- Record transition events between sites
- Export comprehensive selectivity data

## Mathematical Formulation

### Weight Calculation

For each nucleophilic site `i`, a weight `w_i` is calculated based on the selected mode:

#### Distance Mode
```
w_i = exp(-α * d(Nu_i, C)) / Σ_j exp(-α * d(Nu_j, C))
```

#### Geometry Mode
```
quality_i = cos²(θ_i - 180°) * exp(-β * (d(Nu_i, C) - d_target)²)
w_i = quality_i / Σ_j quality_j
```

#### Hybrid Mode
```
w_i = 0.5 * w_distance + 0.5 * w_geometry
```

### Force Application

- **Primary site** (highest weight): `F = w * F_base`
- **Secondary sites**: `F = w * F_base * standby_multiplier`

## Configuration

### Basic Setup

```fnl
restraints {
  multi_rna_attack {
    type = multi_site_backside
    
    # Required parameters
    nucleophiles = [15, 23, 31, 45]  # List of nucleophile indices
    carbon = 5                        # Electrophilic carbon
    leaving_group = 12                # Optional - auto-detected if not specified
    
    # Restraint parameters
    target = 180.0                    # Target angle (degrees)
    base_angle_force_constant = 5.0   # Base force constant for angle
    
    # Optional distance restraint
    target_distance = 2.5             # Target Nu-C distance (Å)
    base_distance_force_constant = 3.0
    
    # Weighting parameters
    weighting_mode = hybrid           # "distance", "geometry", or "hybrid"
    distance_alpha = 0.5              # Distance decay factor
    standby_multiplier = 0.3          # Force reduction for non-primary sites
    
    # Tracking
    write_selectivity = yes
    selectivity_output = selectivity.dat
  }
}
```

## Output Format

The selectivity tracking file contains:

```
# step  primary  RC_15  angle_15  dist_15  weight_15  RC_23  angle_23  dist_23  weight_23  ...  total_E
0       15       -0.5   178.2     2.3      0.45       0.8    165.3     3.1      0.25       ...  12.34
100     15       -0.8   179.1     2.1      0.48       0.6    168.5     2.9      0.28       ...  11.56
200     23       0.2    172.3     2.8      0.32       -0.3   177.8     2.4      0.43       ...  10.89
```

Where:
- `step`: Simulation timestep
- `primary`: Index of the current primary nucleophile
- `RC_X`: Reaction coordinate for site X (d(Nu-C) - d(C-LG))
- `angle_X`: Nu-C-LG angle for site X (degrees)
- `dist_X`: Nu-C distance for site X (Å)
- `weight_X`: Current weight for site X
- `total_E`: Total restraint energy

## Analysis Tools

### Selectivity Metrics

1. **Site Occupancy**: Percentage of time each site is the primary site
2. **Average Reaction Coordinate**: Mean RC value for each site
3. **Transition Matrix**: Frequency of transitions between sites
4. **Geometry Quality Score**: Average angle deviation from 180°

### Python Analysis Script

```python
import pandas as pd
import numpy as np

def analyze_selectivity(filename):
    """Analyze multi-site selectivity data."""
    
    # Read data
    df = pd.read_csv(filename, delim_whitespace=True, comment='#')
    
    # Calculate site occupancy
    n_sites = len([c for c in df.columns if 'RC_' in c])
    site_ids = [int(c.split('_')[1]) for c in df.columns if 'RC_' in c]
    
    occupancy = {}
    for site in site_ids:
        occupancy[site] = (df['primary'] == site).sum() / len(df) * 100
    
    # Find best performing site
    best_site = max(occupancy, key=occupancy.get)
    
    print(f"Site Occupancy (% time as primary):")
    for site, occ in occupancy.items():
        print(f"  Site {site}: {occ:.1f}%")
    
    print(f"\nMost favored site: {best_site}")
    
    # Calculate average reaction coordinates
    print("\nAverage Reaction Coordinates:")
    for site in site_ids:
        rc_col = f'RC_{site}'
        avg_rc = df[rc_col].mean()
        print(f"  Site {site}: {avg_rc:.3f} Å")
    
    return df, occupancy

# Usage
df, occupancy = analyze_selectivity('selectivity_data.dat')
```

## Best Practices

### Parameter Selection

1. **Force Constants**: Start with weak forces (2-5) to allow natural competition
2. **Standby Multiplier**: Use 0.2-0.4 to maintain some restraint on all sites
3. **Distance Alpha**: 0.3-0.7 works well for most systems
4. **Weighting Mode**: 
   - Use "distance" for initial approach simulations
   - Use "geometry" when sites are at similar distances
   - Use "hybrid" for most realistic behavior

### System Setup

1. **Initial Configuration**: Place warhead equidistant from sites initially
2. **Equilibration**: Run without restraints first to relax the system
3. **Production**: Apply multi-site restraints gradually

### Troubleshooting

- **One site always wins**: Reduce standby_multiplier or use weaker forces
- **No clear preference**: Increase force constants or adjust weighting mode
- **Unstable dynamics**: Reduce force constants or increase standby_multiplier

## Example Applications

### RNA 2'-OH Selectivity
```fnl
# Study selectivity between multiple 2'-OH groups
nucleophiles = [O2'_G15, O2'_A23, O2'_U31, O2'_C45]
weighting_mode = geometry  # All 2'-OH at similar distances
```

### Protein Cysteine Competition
```fnl
# Multiple cysteines competing for Michael acceptor
nucleophiles = [SG_C25, SG_C42, SG_C78, SG_C95]
weighting_mode = hybrid  # Consider both distance and orientation
```

### DNA Base Alkylation
```fnl
# N7 vs N3 selectivity in purine bases
nucleophiles = [N7_G10, N3_G10, N7_A15, N3_A15]
weighting_mode = distance  # Initial approach dominated by proximity
```

## Comparison with Standard Backside Restraint

| Feature | Standard | Multi-Site |
|---------|----------|------------|
| Number of nucleophiles | 1 | Multiple |
| Force application | Single site | All sites (weighted) |
| Site selection | Fixed | Dynamic |
| Selectivity tracking | No | Yes |
| Competition modeling | N/A | Soft competition |
| Use case | Single reaction | Selectivity studies |

## References

- SN2 reaction dynamics and selectivity
- RNA nucleophilic sites and reactivity
- Covalent inhibitor design principles
- Competitive nucleophilic substitution mechanisms