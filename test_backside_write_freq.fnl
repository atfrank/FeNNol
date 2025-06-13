
calculation:
  type: md
  steps: 100
  
restraints:
  backside_sn2:
    type: backside_attack
    nucleophile: 0
    carbon: 1
    leaving_group: 2
    target_angle: 180
    angle_force_constant: 0.1
    target_distance: [4.0, 3.5, 3.0, 2.5, 2.0]
    distance_force_constant: 0.05
    write_realtime_pmf: true
    write_frequency: 10  # Write every 10 steps instead of every step
    output_dir: test_output
