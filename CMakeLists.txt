cmake_minimum_required(VERSION 3.18)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
project(cuda_test)
set(CMAKE_CUDA_STANDARD 14)
########## opencv c++ partion##############
set(CMAKE_CXX_STANDARD  11)
set(CMAKE_PREFIX_PATH   /home/maxwell/DJY/opencv3411/debug)
find_package(OpenCV REQUIRED)
# find zed and cuda head file (zed need cuda head)
set(INC_DIR
        "/usr/local/zed/include"
        "/usr/local/cuda-10.2/include"
        )
set(LINK_DIR
        "/usr/local/zed/lib"
        "/usr/local/cuda-10.2/lib64")
set(zed_lib_list
        libsl_ai.so
        libsl_zed.so
        )
# include path
include_directories(${OpenCV_INCLUDE_DIRS} ${INC_DIR})
# lib path
link_directories(${LINK_DIR})
############## End ######################
INCLUDE(/usr/local/share/cmake-3.18/Modules/FindCUDA.cmake)
CUDA_ADD_EXECUTABLE(cuda_test
        cuda/main.cpp cuda/cudademo.cu
        opencv/census.cpp opencv/census.h opencv/devicequery.cpp opencv/devicequery.h
        opencv/disp_main.cpp opencv/disp_main.h
        opencv/dispart_estimate.cpp opencv/dispart_estimate.h
        opencv/image_process_utl.cpp opencv/image_process_utl.h)
# link lib
target_link_libraries(cuda_test ${OpenCV_LIBS} ${zed_lib_list})
#set_target_properties(
#        cuda_test
#        PROPERTIES
#        CUDA_SEPARABLE_COMPILATION ON)