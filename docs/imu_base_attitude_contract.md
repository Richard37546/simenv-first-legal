# IMU/base attitude contract

The startup supervisor subscribes only to `/trunk_imu` (`sensor_msgs/Imu`).
The message frame is `imu_link`.  The xacro chain is identity at both fixed
joints: `base -> trunk` and `trunk -> imu_link`.  The Gazebo IMU plugin also
uses `bodyName=imu_link`, `frameName=imu_link`, and zero RPY offset.

Therefore the offline base-frame conversion is identity for this model:

`q_world_base = q_world_imu`

The ROS message quaternion is read by field name `(x,y,z,w)`.  `IOROS` stores
the same samples internally as `(w,x,y,z)` without a further rotation.

For this audit, upright is evaluated without relying on Euler angles:

`upright_score = dot(z_base_in_world, z_world) = 1 - 2(x^2+y^2)`.

`+1` is upright, `0` is side-on, and `-1` is inverted.  Roll/pitch are only
display values.  Historical keyboard FixedStand samples and new B/C service
samples show matching truth and IMU upright scores near `+1`; a roll near
`-pi` is consequently an inversion/fall indicator in this configured frame,
not an alternative zero-angle convention.
