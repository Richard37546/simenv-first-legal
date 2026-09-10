# FixedStand Runtime Control Contract

The runtime contract for this audit is: global `/robot_name=a1`; one `junior_ctrl`; one publisher per `/a1_gazebo/<joint>_controller/command`; matching controller subscriber; progressing `/clock`; finite continuous servo states; finite normalized `/trunk_imu`; one zero `/cmd_vel` source; and a FixedStand request only after those conditions hold.

`IOROS` resolves `/robot_name` globally, then constructs all twelve state and command topics from it. The launch node `state_from_gazebo` has a private `robot_name` parameter for model lookup and does not configure `IOROS`.

Truth is collected only in the offline audit capture. It is never a service-request, health, planner, or controller input.
