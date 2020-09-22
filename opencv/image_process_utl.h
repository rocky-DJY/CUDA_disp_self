//
// Created by maxwell on 9/8/20.
//
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#ifndef CUDA_TEST_IMAGE_PROCESS_UTL_H
#define CUDA_TEST_IMAGE_PROCESS_UTL_H
#endif //CUDA_TEST_IMAGE_PROCESS_UTL_H
void MedianFilter(const float* in, float* out,
        const int32_t& width, const int32_t& height,const int32_t wnd_size);