cmake_minimum_required(VERSION 3.5)
project(boundary_evaluation)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(PCL REQUIRED COMPONENTS common io filters segmentation)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS random)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Link directories
link_directories(${PCL_LIBRARY_DIRS} ${CERES_LIBRARY_DIRS})

# Add definitions
add_definitions(${PCL_DEFINITIONS})

# Executables
add_executable(boundary_evaluation src/main_6circle.cpp)
add_executable(boundary_evaluation_ver2 src/main_6circle_variable_R.cpp)

# Link libraries
target_link_libraries(boundary_evaluation
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

target_link_libraries(boundary_evaluation_ver2
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  ${Boost_LIBRARIES}
  ${OpenCV_LIBS}
  Eigen3::Eigen
)

# Output directory
set_target_properties(boundary_evaluation boundary_evaluation_ver2 PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/)

