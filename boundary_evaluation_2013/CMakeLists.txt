cmake_minimum_required(VERSION 3.0.2)
project(boundary_evaluation_2013)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  pcl_conversions
  pcl_ros
)

## Find PCL
find_package(PCL 1.8 REQUIRED)

## Declare a catkin package
catkin_package(
  CATKIN_DEPENDS roscpp sensor_msgs pcl_conversions pcl_ros
)

## Specify additional locations of header files
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

## Declare the executable and name it 'filtered_boundary_points'
add_executable(filtered_boundary_points src/filtered_boundary_points.cpp)

## Add dependencies to the executable
add_dependencies(filtered_boundary_points ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link the executable target against
target_link_libraries(filtered_boundary_points
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

## Install the executable
install(TARGETS filtered_boundary_points
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

