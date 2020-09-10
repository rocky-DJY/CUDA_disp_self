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
void MedianFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize);