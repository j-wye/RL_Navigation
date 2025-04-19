cmake_minimum_required(VERSION 3.5)
project(rl_navigation)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz2 REQUIRED)

install(
  DIRECTORY
  config
  launch
  scripts
  usd
  DESTINATION share/${PROJECT_NAME}
)

ament_package()