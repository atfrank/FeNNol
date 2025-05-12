device cuda:0
model_file ../ani2x.fnx

traj_format arc
per_atom_energy no
energy_unit Ha

xyz_input{
  file aspirin.xyz 
  # whether the first column is the atom index (Tinker format)
  indexed no 
  # whether a comment line is present
  has_comment_line yes
}

# number of steps to perform
nsteps = 1000000
# timestep of the dynamics
dt[fs] = 0.5 

#time between each saved frame
tdump[ps] = 1.
# number of steps between each printing of the energy
nprint = 100
nsummary = 1000

## set the thermostat
#thermostat NVE
thermostat LGV 

## Thermostat parameters
temperature = 300.
#friction constant
gamma[THz] = 1.

## Restraints section
restraints {
  # Harmonic restraint between atoms 0 and 7 (C-O bond)
  co_bond {
    type = distance
    atom1 = 0
    atom2 = 7
    target = 1.3
    force_constant = 20.0
    style = harmonic
  }
  
  # Flat-bottom restraint to keep angle between atoms 5, 12, 7 within tolerance
  cco_angle {
    type = angle
    atom1 = 5
    atom2 = 12
    atom3 = 7
    target = 2.1
    force_constant = 10.0
    style = flat_bottom
    tolerance = 0.2
  }
}

## Collective variables to track
colvars {
  c_o_dist {
    type = distance
    atom1 = 0
    atom2 = 7
  }
  c_c_o_angle {
    type = angle
    atom1 = 5
    atom2 = 12
    atom3 = 7
  }
}