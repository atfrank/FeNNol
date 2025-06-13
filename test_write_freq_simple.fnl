calculation:
  type: md
  steps: 50
  timestep: 0.5
  
system:
  positions_file: aspirin.arc
  
potential:
  type: gap
  
integrator:
  type: verlet
  temperature: 300
  
restraints:
  test_backside:
    type: backside_attack
    nucleophile: 0
    carbon: 1
    leaving_group: 2
    target_angle: 180
    angle_force_constant: 0.1
    target_distance: [3.0, 2.5]
    distance_force_constant: 0.05
    write_realtime_pmf: true
    write_frequency: 10
    output_dir: .