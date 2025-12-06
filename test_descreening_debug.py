#!/usr/bin/env python3
"""Debug descreening integral calculation."""

import numpy as np
import jax.numpy as jnp

# Test with O-H pair from water molecule
r = 0.957311339  # O-H distance
rho_i = 1.5  # O radius
rho_j = 1.3  # H radius

print(f"Testing descreening integral for O-H pair:")
print(f"  r = {r:.6f} Å")
print(f"  rho_i (O) = {rho_i} Å")
print(f"  rho_j (H) = {rho_j} Å")
print()

# Compute bounds
upper_limit = rho_i + rho_j
lower_limit = abs(rho_i - rho_j)

print(f"Bounds:")
print(f"  lower_limit = |{rho_i} - {rho_j}| = {lower_limit}")
print(f"  upper_limit = {rho_i} + {rho_j} = {upper_limit}")
print(f"  r < lower_limit? {r < lower_limit}")
print(f"  r < upper_limit? {r < upper_limit}")
print()

# Since r (0.957) < upper_limit (2.8) but r > lower_limit (0.2),
# we're in the partial overlap region
print("Region: PARTIAL OVERLAP (use actual r)")
print()

r_for_calc = r if r >= lower_limit else lower_limit
s_j = rho_j

# Compute l_ij = 1 / max(rho_i, |r - s_j|)
abs_diff = abs(r_for_calc - s_j)
lower_bound = max(rho_i, abs_diff)
l_ij = 1.0 / lower_bound

print(f"Lower bound calculation:")
print(f"  |r - s_j| = |{r_for_calc} - {s_j}| = {abs_diff}")
print(f"  lower_bound = max({rho_i}, {abs_diff}) = {lower_bound}")
print(f"  l_ij = 1 / {lower_bound} = {l_ij}")
print()

# Compute u_ij = 1 / (r + s_j)
u_ij = 1.0 / (r_for_calc + s_j)
print(f"Upper bound calculation:")
print(f"  u_ij = 1 / ({r_for_calc} + {s_j}) = {u_ij}")
print()

# Precompute terms
l_ij2 = l_ij * l_ij
u_ij2 = u_ij * u_ij
s_j2 = s_j * s_j
r_inv = 1.0 / r_for_calc
ratio = np.log(u_ij / l_ij)

print(f"HCT formula components:")
print(f"  l_ij² = {l_ij2}")
print(f"  u_ij² = {u_ij2}")
print(f"  s_j² = {s_j2}")
print(f"  1/r = {r_inv}")
print(f"  ln(u_ij/l_ij) = {ratio}")
print()

# HCT integral formula
term1 = l_ij - u_ij
term2 = 0.25 * r_for_calc * (u_ij2 - l_ij2)
term3 = 0.5 * r_inv * ratio
term4 = 0.25 * s_j2 * r_inv * (l_ij2 - u_ij2)

print(f"HCT formula terms:")
print(f"  term1 = l_ij - u_ij = {term1}")
print(f"  term2 = 0.25*r*(u_ij² - l_ij²) = {term2}")
print(f"  term3 = 0.5*ln(u_ij/l_ij)/r = {term3}")
print(f"  term4 = 0.25*s_j²/r*(l_ij² - u_ij²) = {term4}")
print()

integral = term1 + term2 + term3 + term4
print(f"Total descreening integral: {integral}")
print()

# Now compute what CUDA should give
print("="*60)
print("Expected from CUDA descreening integral for this pair:")
print()

# For water molecule with 3 atoms, compute psi_sum for oxygen
# Oxygen has two H neighbors
psi_O_from_H1 = integral  # Same as above
psi_O_from_H2 = integral  # Symmetric
psi_O_total = psi_O_from_H1 + psi_O_from_H2

print(f"For oxygen atom:")
print(f"  ψ(O, H1) = {psi_O_from_H1}")
print(f"  ψ(O, H2) = {psi_O_from_H2}")
print(f"  ψ_sum(O) = {psi_O_total}")
print()

# OBC formula: 1/R_i = 1/ρ_i - tanh(ψ - b*ψ² + c*ψ³) / ρ_i
b_O = 0.85
c_O = 0.1
psi = psi_O_total

tanh_arg = psi - b_O * psi**2 + c_O * psi**3
tanh_val = np.tanh(tanh_arg)

print(f"OBC Born radius calculation for oxygen:")
print(f"  ψ = {psi}")
print(f"  tanh_arg = ψ - b*ψ² + c*ψ³ = {tanh_arg}")
print(f"  tanh(arg) = {tanh_val}")
print(f"  1/R_O = 1/{rho_i} - {tanh_val}/{rho_i} = {1/rho_i - tanh_val/rho_i}")
print(f"  R_O = {1.0 / (1/rho_i - tanh_val/rho_i)}")
print()

# For hydrogen (symmetric, each H sees one O and one H)
r_HH = 1.514  # H-H distance
rho_H = 1.3

# H-O interaction
upper_HO = rho_H + rho_i
lower_HO = abs(rho_H - rho_i)
print(f"For hydrogen-oxygen pair (r={r}):")
print(f"  Same as above, integral = {integral}")

# H-H interaction
upper_HH = rho_H + rho_H
lower_HH = abs(rho_H - rho_H)
print(f"\nFor hydrogen-hydrogen pair (r={r_HH}):")
print(f"  upper_limit = {upper_HH}")
print(f"  lower_limit = {lower_HH}")

r_HH_calc = r_HH if r_HH >= lower_HH else lower_HH
s_j_HH = rho_H
abs_diff_HH = abs(r_HH_calc - s_j_HH)
lower_bound_HH = max(rho_H, abs_diff_HH)
l_ij_HH = 1.0 / lower_bound_HH
u_ij_HH = 1.0 / (r_HH_calc + s_j_HH)

l_ij2_HH = l_ij_HH * l_ij_HH
u_ij2_HH = u_ij_HH * u_ij_HH
s_j2_HH = s_j_HH * s_j_HH
r_inv_HH = 1.0 / r_HH_calc
ratio_HH = np.log(u_ij_HH / l_ij_HH)

integral_HH = (l_ij_HH - u_ij_HH +
               0.25 * r_HH_calc * (u_ij2_HH - l_ij2_HH) +
               0.5 * r_inv_HH * ratio_HH +
               0.25 * s_j2_HH * r_inv_HH * (l_ij2_HH - u_ij2_HH))

print(f"  H-H integral = {integral_HH}")

psi_H_total = integral + integral_HH
b_H = 0.85
c_H = 0.72

tanh_arg_H = psi_H_total - b_H * psi_H_total**2 + c_H * psi_H_total**3
tanh_val_H = np.tanh(tanh_arg_H)

R_H = 1.0 / (1/rho_H - tanh_val_H/rho_H)
print(f"\nOBC Born radius for hydrogen:")
print(f"  ψ_sum = {psi_H_total}")
print(f"  R_H = {R_H}")
