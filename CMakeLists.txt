cmake_minimum_required(VERSION 3.1)
project(mrs_optic_flow)

FILE(GLOB_RECURSE HEADER_FILES include *.h)
FILE(GLOB_RECURSE OPENCL_FILES include *.cl)
FILE(GLOB_RECURSE LAUNCHER_FILES include *.launch)
FILE(GLOB_RECURSE YAML_FILES include *.yaml)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
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

# set(CMAKE_CXX_FLAGS "-std=c++17 -fno-diagnostics-color -O3 -march=native")

# include Eigen3
find_package(Eigen3 REQUIRED)
set(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})
set(Eigen_LIBRARIES ${Eigen_LIBRARIES})

find_package(OpenCV REQUIRED)
# find_package(OpenCL REQUIRED)
# message("${OpenCL_INCLUDE_DIRS} ${OpenCL_LIBRARIES}")

###################################
## catkin specific configuration ##
###################################

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES fft_method # block_method  fastSpaced_method
  CATKIN_DEPENDS mrs_lib cv_bridge image_transport roscpp sensor_msgs std_msgs geometry_msgs message_runtime
  #  DEPENDS system_lib
  DEPENDS Eigen
  LIBRARIES OpticFlow
  )

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
  #  ${OpenCL_INCLUDE_DIRS}
  )

## Declare a C++ library
add_library(util_functions src/utilityFunctions.cpp)
# add_library(block_method src/BlockMethod.cpp)
add_library(fft_method src/FftMethod.cpp)
# # add_custom_target(fft_method_cl COMMAND "pwd; cp src/FftMethod.cl ~/home/.ros/" SOURCES src/FftMethod.cl)
# add_custom_target(fft_method_cl SOURCES src/FftMethod.cl)
# add_custom_command(TARGET fft_method_cl PRE_BUILD
#                    COMMAND ${CMAKE_COMMAND} -E copy
#                        ${CMAKE_SOURCE_DIR}/src/FftMethod.cl ~/.ros/)
add_library(scaleRotationEstimator src/scaleRotationEstimator.cpp)
#add_library(fastSpacedOCL_method src/FastSpacedBMMethod_OCL.cpp)

target_link_libraries(fft_method ${OpenCV_LIBRARIES})
# target_link_libraries(block_method ${OpenCV_LIBRARIES})
target_link_libraries(scaleRotationEstimator ${OpenCV_LIBRARIES})
#target_link_libraries(fastSpacedOCL_method ${OpenCV_LIBRARIES} ${OpenCL_LIBRARIES} ${catkin_LIBRARIES})

## Declare a C++ executable
add_library(OpticFlow src/optic_flow.cpp)


## Add cmake target dependencies of the executable
## same as for the library above
add_dependencies(OpticFlow
  ${${PROJECT_NAME}_EXPORTED_TARGETS} 
  ${catkin_EXPORTED_TARGETS}
  # fft_method_cl
  )

add_custom_target(headers SOURCES ${HEADER_FILES} ${OPENCL_FILES} ${LAUNCHER_FILES} ${YAML_FILES})
#configure_file(src/FastSpacedBMMethod.cl ${CATKIN_DEVEL_PREFIX}/${CATKIN_PACKAGE_BIN_DESTINATION}/FastSpacedBMMethod.cl COPYONLY)

#target_link_libraries(fastSpaced_method fsbmof)

## Specify libraries to link a library or executable target against
target_link_libraries(OpticFlow
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${Eigen_LIBRARIES}
  # block_method
  fft_method
  # fastSpacedOCL_method
  util_functions
  scaleRotationEstimator
  )
