cmake_minimum_required(VERSION 3.1)
project(mrs_optic_flow)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  nodelet
  cv_bridge
  image_transport
  sensor_msgs
  std_msgs
  geometry_msgs
  tf
  mrs_lib
  )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

find_package(OpenCV REQUIRED)

execute_process(COMMAND lsb_release -cs
  OUTPUT_VARIABLE RELEASE_CODENAME
  OUTPUT_STRIP_TRAILING_WHITESPACE
  )

# find_package(OpenCL REQUIRED)
# message("${OpenCL_INCLUDE_DIRS} ${OpenCL_LIBRARIES}")

###################################
## catkin specific configuration ##
###################################

set(LIBRARIES
  OpticFlow
  )

catkin_package(
  INCLUDE_DIRS include
  CATKIN_DEPENDS mrs_lib cv_bridge image_transport roscpp sensor_msgs std_msgs geometry_msgs
  DEPENDS Eigen
  LIBRARIES ${LIBRARIES}
  )

## --------------------------------------------------------------
## |                            Build                           |
## --------------------------------------------------------------

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  )

## Declare a C++ executable
add_library(OpticFlow
  src/optic_flow.cpp
  src/utilityFunctions.cpp
  src/FftMethod.cpp
  src/scaleRotationEstimator.cpp)

if (${RELEASE_CODENAME} MATCHES "bionic")
  target_compile_definitions(OpticFlow PUBLIC ROS_MELODIC=1)
elseif(${RELEASE_CODENAME} MATCHES "focal")
  target_compile_definitions(OpticFlow PUBLIC ROS_NOETIC=1)
endif()

add_dependencies(OpticFlow
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  )

# add_custom_target(headers SOURCES ${HEADER_FILES} ${OPENCL_FILES} ${LAUNCHER_FILES} ${YAML_FILES})
# configure_file(src / FastSpacedBMMethod.cl ${CATKIN_DEVEL_PREFIX} / ${CATKIN_PACKAGE_BIN_DESTINATION} / FastSpacedBMMethod.cl COPYONLY)

target_link_libraries(OpticFlow
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Eigen_LIBRARIES}
  )

## --------------------------------------------------------------
## |                           Install                          |
## --------------------------------------------------------------

install(TARGETS ${LIBRARIES}
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
  )

install(DIRECTORY launch config cl
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  )

install(DIRECTORY include/
  DESTINATION ${CATKIN_GLOBAL_INCLUDE_DESTINATION}
  )

install(DIRECTORY ./
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.xml"
  )
