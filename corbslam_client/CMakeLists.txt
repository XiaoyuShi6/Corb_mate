cmake_minimum_required(VERSION 2.8)
project(corbslam_client)
#project(ORB_SLAM2)


IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native ")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        tf
        image_transport
        )

find_package(OpenCV 3.0 QUIET)
if(NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if(NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif()
endif()
find_package(Eigen3 REQUIRED)
find_package(Pangolin REQUIRED)

### find and configure Boost
find_package(Boost COMPONENTS serialization system filesystem REQUIRED )
find_package(PCL 1.7 REQUIRED)


## Generate messages in the 'msg' folder
add_message_files(
        FILES
        corbslam_message.msg
)

## Generate services in the 'srv' folder
add_service_files(
        FILES
        corbslam_insert.srv
        corbslam_update.srv
)

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
generate_messages(
        DEPENDENCIES
        std_msgs
)

catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES orbslam_client
         # CATKIN_DEPENDS roscpp rospy std_msgs
        #  DEPENDS system_lib
)


include_directories(
        ${catkin_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${Boost_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}/Examples/RGB-D
        ${PCL_INCLUDE_DIRS}
        #/usr/include/pcl-1.7
        /opt/ros/kinetic/include/cv_bridge
)
#add_definitions( ${PCL_DEFINITIONS} )
#link_directories( ${PCL_LIBRARY_DIRS} )
## Declare a C++ library

LINK_DIRECTORIES(/usr/local/lib )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/KeyFrame.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        include/Cache.h
        src/Cache.cc
        include/LightKeyFrame.h
        src/LightKeyFrame.cc
        include/LightMapPoint.h
        src/LightMapPoint.cc
        include/SerializeObject.h
        include/DataDriver.h
        src/DataDriver.cc
        include/TransPose.h src/TransPose.cpp
        src/Pointcloudmapping.cpp
        )

target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${catkin_LIBRARIES}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${Boost_LIBRARIES}
        ${PCL_LIBRARIES}
        /opt/ros/kinetic/lib/libcv_bridge.so #cv_bridge的so文件
        )


## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(orbslam_client ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)

add_executable(corbslam_client_rgbd_tum
        Examples/RGB-D/rgbd_tum.cc
        Examples/RGB-D/ros_viewer.cpp
        )

target_link_libraries(corbslam_client_rgbd_tum
        ${catkin_LIBRARIES}
        ${PROJECT_NAME}
        )
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/ros_test)
add_executable(ros_RGBD
       ros_test/ros_rgbd.cc)
target_link_libraries(ros_RGBD
        ${catkin_LIBRARIES}
        ${PROJECT_NAME})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Stereo)

add_executable(corbslam_client_stereo_kitti
        Examples/Stereo/stereo_kitti.cc)
target_link_libraries(corbslam_client_stereo_kitti
        ${catkin_LIBRARIES}
        ${PROJECT_NAME})

#add_executable(orbslam_client_node_stereo_euroc
#        Examples/Stereo/stereo_euroc.cc)
#target_link_libraries(orbslam_client_node_stereo_euroc ${PROJECT_NAME})
#


#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
#
#add_executable(corbslam_client_mono_tum
#        Examples/Monocular/mono_tum.cc)
#
#target_link_libraries(corbslam_client_mono_tum
#        ${PROJECT_NAME}
#        ${catkin_LIBRARIES}
#        )

#add_executable(orbslam_client_node_mono_kitti
#        Examples/Monocular/mono_kitti.cc)
#target_link_libraries(orbslam_client_node_mono_kitti ${PROJECT_NAME})
#
#add_executable(orbslam_client_node_mono_euroc
#        Examples/Monocular/mono_euroc.cc)
#target_link_libraries(orbslam_client_node_mono_euroc ${PROJECT_NAME})

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS corbslam_client
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
#
## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_orbslam_client.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
