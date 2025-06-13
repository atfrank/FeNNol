device cuda:0 
enable_x64
matmul_prec highest
print_timings yes
restraint_debug true
nreplicas 1

model_file ../mace_mp_medium.fnx

xyz_input{
  file sn2_optimized_mg.xyz
  indexed no
  has_comment_line yes
}

minimum_image no
wrap_box no
estimate_pressure no

nsteps = 12000
dt[fs] = 0.5
traj_format xyz

nblist_skin 1.0

tdump[ps] = 0.005
nprint = 10
nsummary = 100
nblist_verbose

thermostat ADQTB 

temperature = 150.
gamma[THz] = 1.

qtb{
  tseg[ps]=0.25
  omegacut[cm1]=15000.
  skipseg = 5
  startsave = 50
  agamma = 1.
}

restraints {
   sn2_auto {
      type = adaptive_sn2
      nucleophile = 119
      carbon = 536
      leaving_group = 540
      simulation_steps = 12000  # FIXED: Must match nsteps
    }

  keep_restraint1 {
    type = distance
    atom1 = 119
    atom2 = 536
    target = 4.0
    force_constant = 0.0
    style = flat_bottom
    tolerance = 0.2
    # REMOVED: update_steps not needed for non-time-varying restraints
  }
  
  keep_restraint2 {
    type = distance
    atom1 = 535
    atom2 = 540
    target = 1.3
    force_constant = 10.0
    style = flat_bottom
    tolerance = 0.2
    # REMOVED: update_steps not needed for non-time-varying restraints
  }
  
  keep_restraint3 {
    type = distance
    atom1 = 536
    atom2 = 535
    target = 1.3
    force_constant = 10.0
    style = flat_bottom
    tolerance = 0.2
    # REMOVED: update_steps not needed for non-time-varying restraints
  }
}