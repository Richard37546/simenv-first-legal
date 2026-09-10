# A1 Control Startup Contract

The only safe startup sequence established by the V1 audit is: launch Gazebo and `junior_ctrl`; as soon as the controller has published baseline `rl_mode_ready=false` and joint command topics are connected, send `2`; wait for the FixedStand transition; only then send `6` and maintain zero `/cmd_vel`.

The initial FSM state is `PASSIVE`. In that state simulation motor gains are passive, so a gravity-loaded model must not be held there while unrelated perception readiness waits consume simulation time. FixedStand transitions to `[0, 0.67, -1.3]` for each controller-order leg triple over 1000 controller steps. RL then runs `act_inference` every 0.02 s and maps `q = 0.25 * action[reindex] + default_dof_pos[reindex]`, with `Kp=80`, `Kd=1`.

The controller order is `FR, FL, RR, RL`; policy order is remapped by `[3,4,5,0,1,2,9,10,11,6,7,8]`. The source has no action clamp. `trunk_imu` is in `imu_link` and is copied as quaternion `w,x,y,z`.

`/unitree/rl_mode_ready` is valid only with an external receipt-time TTL. The Bool topic is unstamped and latched, so process death cannot itself produce a new false message.
