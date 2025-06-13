device cuda:0 
enable_x64
matmul_prec highest
print_timings yes
restraint_debug true
nreplicas 1

model_file ./mace_mp_medium.fnx

xyz_input{
  file sub_system_premodified_epoxide_minimized_solv_Mg_H+.xyz
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
nsteps = 20000
# timestep of the dynamics
dt[fs] =  .5
traj_format xyz

nblist_skin 1.0

#time between each saved frame
tdump[ps] = 0.05
# number of steps between each printing of the energy
nprint = 10
nsummary = 100
nblist_verbose

## set the thermostat
#thermostat  NVE 
#thermostat  NOSE 
#thermostat LGV 
thermostat ADQTB 

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

# Restraints section - FIXED: Need named restraint blocks
restraints {
  backside_attack_sn2 {
    type = backside_attack
    nucleophile = 119
    carbon = 536
    leaving_group = 540
    
    # Angle component (always present)
    target = 3.14159                # 180° backside approach
    angle_force_constant = 0.1 2.0 10.0 20.0      # Gradual angle control
    
    # Distance component (optional)
    target_distance = 2.5           # Nucleophile-carbon distance in Å
    distance_force_constant = 15.0 12.0 8.0 2.0   # Strong initial pull, then relax
      
    update_steps = 5000
    style = harmonic
  }
}

# Collective variables to track
colvars {
  bonding_restraint_distance {
    type = distance
    atom1 = 119
    atom2 = 536  
  } 
  breaking_restraint_distance {
    type = distance
    atom1 = 540
    atom2 = 536  
  } 
  keep1_restraint_distance {
    type = distance
    atom1 = 535
    atom2 = 536  
  } 
  keep2_restraint_distance {
    type = distance
    atom1 = 535
    atom2 = 540  
  } 
  keep3_restraint_distance {
    type = distance
    atom1 = 535
    atom2 = 534  
  } 
  back_side_1_distance {
    type = distance
    atom1 = 119  
    atom2 = 536
  } 
  back_side_2_distance {
    type = distance
    atom1 = 119  
    atom2 = 535
  } 
  back_side_3 {
    type = distance
    atom1 = 119  
    atom2 = 540
  } 
  back_side_1_angle {
    type = angle
    atom1 = 540  
    atom2 = 536
    atom3 = 119   
  } 
  back_side_2_angle {
    type = angle
    atom1 = 540  
    atom2 = 536
    atom3 = 535   
  } 
}