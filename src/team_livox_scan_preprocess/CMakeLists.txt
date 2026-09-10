cmake_minimum_required(VERSION 3.0.2)
project(team_livox_scan_preprocess)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  std_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/l1s_scan_clean_cloud_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
