# HCT Descreening Derivative Formula

## OpenMM Born Sum Formula

From OpenMM's `gbsaObc.cc`:

```
ψ = l_ij - u_ij + 0.5/r * ln(u_ij/l_ij) + 0.25*r*(u_ij² - l_ij²) + 0.25*(s_j²/r)*(l_ij² - u_ij²)
```

Where:
- `l_ij = 1/max(ρ_i, |r - s_j|)`  (lower bound reciprocal)
- `u_ij = 1/(r + s_j)`            (upper bound reciprocal)
- `s_j = scaledRadiusJ` (atomic radius of j, possibly with scaling)
- `r` = distance between atoms i and j

## Taking the Derivative ∂ψ/∂r

Let's compute each term:

### Term 1: l_ij
For partial overlap (r > |ρ_i - s_j|), we have `max(ρ_i, |r - s_j|) = r - s_j`

So: `l_ij = 1/(r - s_j)`

And: `∂l_ij/∂r = -1/(r - s_j)² = -l_ij²`

### Term 2: u_ij
`u_ij = 1/(r + s_j)`

`∂u_ij/∂r = -1/(r + s_j)² = -u_ij²`

### Term 3: ln(u_ij/l_ij) / (2r)
Using product rule:
```
∂/∂r [ln(u_ij/l_ij)/(2r)] = (1/(2r)) * ∂/∂r[ln(u_ij/l_ij)] + ln(u_ij/l_ij) * ∂/∂r[1/(2r)]

= (1/(2r)) * [∂u_ij/∂r / u_ij - ∂l_ij/∂r / l_ij] + ln(u_ij/l_ij) * (-1/(2r²))

= (1/(2r)) * [-u_ij - (-l_ij)] - ln(u_ij/l_ij)/(2r²)

= (1/(2r)) * (l_ij - u_ij) - ln(u_ij/l_ij)/(2r²)
```

### Term 4: 0.25 * r * (u_ij² - l_ij²)
```
∂/∂r [0.25*r*(u_ij² - l_ij²)] = 0.25*(u_ij² - l_ij²) + 0.25*r * ∂/∂r[u_ij² - l_ij²]

= 0.25*(u_ij² - l_ij²) + 0.25*r * [2*u_ij*(-u_ij²) - 2*l_ij*(-l_ij²)]

= 0.25*(u_ij² - l_ij²) + 0.25*r * [-2*u_ij³ + 2*l_ij³]

= 0.25*(u_ij² - l_ij²) + 0.5*r * (l_ij³ - u_ij³)
```

### Term 5: 0.25 * (s_j²/r) * (l_ij² - u_ij²)
```
∂/∂r [0.25*s_j²/r * (l_ij² - u_ij²)] = 0.25*s_j² * ∂/∂r[(l_ij² - u_ij²)/r]

= 0.25*s_j² * [(∂/∂r[l_ij² - u_ij²])*r - (l_ij² - u_ij²)]/ r²

= 0.25*s_j² * [[2*l_ij*(-l_ij²) - 2*u_ij*(-u_ij²)]*r - (l_ij² - u_ij²)] / r²

= 0.25*s_j² * [2*r*(u_ij³ - l_ij³) - (l_ij² - u_ij²)] / r²
```

## Complete Derivative Formula

```
∂ψ/∂r = -l_ij² + u_ij²
      + (1/(2r))*(l_ij - u_ij) - ln(u_ij/l_ij)/(2r²)
      + 0.25*(u_ij² - l_ij²) + 0.5*r*(l_ij³ - u_ij³)
      + 0.25*s_j²/r² * [2*r*(u_ij³ - l_ij³) - (l_ij² - u_ij²)]
```

Simplifying:
```
∂ψ/∂r = -l_ij² + u_ij² + 0.25*(u_ij² - l_ij²)
      + (1/(2r))*(l_ij - u_ij) - ln(u_ij/l_ij)/(2r²)
      + 0.5*r*(l_ij³ - u_ij³)
      + 0.5*s_j²/r * (u_ij³ - l_ij³)
      - 0.25*s_j²/r² * (l_ij² - u_ij²)

= -0.75*l_ij² + 1.25*u_ij²
  + 0.5/r * (l_ij - u_ij)
  - 0.5/r² * ln(u_ij/l_ij)
  + 0.5*r*(l_ij³ - u_ij³)
  + 0.5*s_j²/r*(u_ij³ - l_ij³)
  - 0.25*s_j²/r²*(l_ij² - u_ij²)
```

Further combining:
```
∂ψ/∂r = -0.75*l_ij² + 1.25*u_ij²
      + 0.5/r * [(l_ij - u_ij) - ln(u_ij/l_ij)/r]
      + 0.5*(l_ij³ - u_ij³)*(r - s_j²/r)
      - 0.25*s_j²/r²*(l_ij² - u_ij²)
```

This is the HCT formula that OpenMM uses!

## Our Current (Wrong) Formula

We currently use the simplified formula:
```
∂ψ/∂r = -ρ_i/r³
```

This is derived from the much simpler integral:
```
ψ_simple = 0.5*ρ_i*(1/r² - 1/upper²)
```

The HCT formula is completely different and accounts for multiple terms with different r-dependencies!
