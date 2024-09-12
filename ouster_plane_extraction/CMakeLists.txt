cmake_minimum_required(VERSION 3.0.2)
project(ouster_plane_extraction)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  pcl_ros
  pcl_conversions
  sensor_msgs
)

find_package(PCL 1.7 REQUIRED)

catkin_package(
  CATKIN_DEPENDS roscpp pcl_ros pcl_conversions sensor_msgs
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_executable(ouster_plane_extraction src/ouster_plane_extraction.cpp)

target_link_libraries(ouster_plane_extraction
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

install(TARGETS ouster_plane_extraction
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

