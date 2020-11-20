cmake_minimum_required(VERSION 3.18)

#set(CMAKE_CUDA_COMPILER /usr/local/cuda-10.2/bin/nvcc)
project(cuda_test)
set(CMAKE_CUDA_STANDARD 14)
## openmp ##
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
########## opencv c++ partion##############
set(CMAKE_CXX_STANDARD  11)
set(OpenCV_DIR   /home/maxwell/DJY/opencv3411/debug/share/OpenCV)
find_package(OpenCV REQUIRED)
# find zed and cuda head file (zed need cuda head)
set(INC_DIR
        "/usr/local/zed/include"
        #"/usr/local/cuda-10.2/include"
        )
set(LINK_DIR
        "/usr/local/zed/lib"
        #"/usr/local/cuda-10.2/lib64"
        )
set(zed_lib_list
        libsl_ai.so
        libsl_zed.so
        )
# include path
include_directories(${OpenCV_INCLUDE_DIRS} ${INC_DIR})
# lib path
link_directories(${LINK_DIR})
############## End ######################
INCLUDE(/usr/local/share/cmake-3.19/Modules/FindCUDA.cmake)
CUDA_ADD_EXECUTABLE(cuda_test
        cuda/main.cpp cuda/aggcuda.cu
        opencv/feature_descript.cpp opencv/feature_descript.h opencv/devicequery.cpp opencv/devicequery.h
        opencv/disp_main.cpp opencv/disp_main.h
        opencv/correTest.cpp opencv/dispart_estimate.h
        opencv/image_process_utl.cpp opencv/image_process_utl.h
        opencv/corre.cpp opencv/corre.h opencv/computeXYZ.cpp opencv/computeXYZ.h
        opencv/read.cpp opencv/read.h
        cuda/census_cuda.cu
        cuda/costcompute.cu
        )
# link lib
target_link_libraries(cuda_test ${OpenCV_LIBS} ${zed_lib_list})