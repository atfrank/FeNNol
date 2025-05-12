device cuda:0 
enable_x64
matmul_prec default
print_timings yes

model_file ./mace_mp_small.fnx

xyz_input{
  file sub_system_premodified_epoxide_minimized_solv.xyz
  # whether the first column is the atom index (Tinker format)
  indexed no
  # whether a comment line is present
  has_comment_line yes
}

#0 cell vectors (format:  ax ay az bx by bz cx cy cz)
#cell = 62.23 0. 0. 0. 62.23 0. 0. 0. 62.23
# compute neighborlists using minimum image convention
minimum_image no
# whether to wrap the atoms inside the first unit cell
wrap_box no
estimate_pressure no

# number of steps to perform
nsteps = 100000
# timestep of the dynamics
dt[fs] =  .5
traj_format xyz

nblist_skin 2.

#time between each saved frame
tdump[ps] = 0.05
# number of steps between each printing of the energy
nprint = 10
nsummary = 100
nblist_verbose

## set the thermostat
#thermostat  NVE 
#thermostat  NOSE 
thermostat LGV 
#thermostat ADQTB 

## Thermostat parameters
temperature = 300.
#friction constant
gamma[THz] = 10.

## parameters for the Quantum Thermal Bath
qtb{
  # segment length for noise generation and spectra
  tseg[ps]=0.25
  # maximum frequency
  omegacut[cm1]=15000.
  # number of segments to skip before adaptation
  skipseg = 5
  # number of segments to wait before accumulation statistics on spectra
  startsave = 50
  # parameter controlling speed of adQTB adaptation 
  agamma  = 1.
}

# Restraints section
restraints {
  # Backbone position restraints on CA atoms
  # These restrain selected alpha carbons to their initial positions
  ca_restraint1 {
    type = distance
    atom1 = 562  # Mg+ ion
    atom2 = 539  # epoxide O
    target = 3.0
    force_constant = 5.0
    style = flat_bottom
    tolerance = 0.5  # Allow movement within 1 Å
  }  
}

# Collective variables to track
colvars {
  end_to_end_distance {
    type = distance
    atom1 = 562  # Mg+ ion
    atom2 = 539  # epoxide O
  } 
}
