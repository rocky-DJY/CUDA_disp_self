//
// Created by lab307 on 8/1/20.
//

#include "devicequery.h"
codetest::codetest() {}
codetest::~codetest() {}
void codetest::test() {
    int dev = 0;
    cudaDeviceProp devProp;
    //CHECK(cudaGetDeviceProperties(&devProp, dev));
    cudaGetDeviceProperties(&devProp, dev);
    std::cout << "使用GPU device " << dev << ": " << devProp.name << std::endl;
    std::cout << "SM的数量：" << devProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << devProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << devProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个SM的最大线程数：" << devProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "SP size per SM ： " << devProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::printf("warp size: %d\r\n", devProp.warpSize);
    //std::printf("max thread size: %d\r\n", (devProp.maxGridSize[1]+1)*devProp.maxThreadsPerBlock);
}