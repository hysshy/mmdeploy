# Copyright (c) OpenMMLab. All rights reserved.

project(mmdeploy_mmaction)

file(GLOB SRCS ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp")
mmdeploy_add_module(${PROJECT_NAME} "${SRCS}")
add_subdirectory(cpu)
add_subdirectory(cuda)
target_link_libraries(${PROJECT_NAME} PRIVATE
    mmdeploy::transform
    mmdeploy_opencv_utils)

add_library(mmdeploy::mmaction ALIAS ${PROJECT_NAME})

set(MMDEPLOY_TASKS ${MMDEPLOY_TASKS} video_recognizer CACHE INTERNAL "")
