cmake_minimum_required(VERSION 3.5)
project(boundary_evaluation)

# C++17 표준 설정
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PCL 패키지 찾기
find_package(PCL 1.8 REQUIRED)

# Eigen 패키지 찾기
find_package(Eigen3 REQUIRED)

# 실행 파일 추가
add_executable(boundary_evaluation src/boundary_find_urop.cpp)

# 라이브러리 링크
target_link_libraries(boundary_evaluation 
  ${PCL_LIBRARIES}
  Eigen3::Eigen
)

# 헤더 파일 경로 추가
include_directories(
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# 빌드 후 바이너리 위치 설정
set_target_properties(boundary_evaluation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/
)

