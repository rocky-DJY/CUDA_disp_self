//
// Created by lab307 on 8/3/20.
//

#ifndef CUDA_TEST_DISP_MAIN_H
#define CUDA_TEST_DISP_MAIN_H
#endif //CUDA_TEST_DISP_MAIN_H

#include "feature_descript.h"
#include "dispart_estimate.h"
#include <iostream>
// ZED includes
#include <sl/Camera.hpp>
// OpenCV includes
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include "devicequery.h"
using namespace sl;
using namespace std;
cv::Mat slMat2cvMat(Mat& input);  // sl到opencv 数据转换
int DispMain();