//
// Created by maxwell on 10/16/20.
//
//////////*************Description*****************///////////////////
//构造函数传入右相机的内参旋转平移矩阵
//调用视差转为为匹配点对的
#ifndef CUDA_TEST_COMPUTEXYZ_H
#define CUDA_TEST_COMPUTEXYZ_H
#include <algorithm>
#include <iterator>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include "image_process_utl.h"
using namespace std;
class computeXYZ {
public:
    computeXYZ(cv::Mat image_left,cv::Mat leftIntrinsic,cv::Mat rightIntrinsic,cv::Mat rightRotation,cv::Mat rightTranslation);
    cv::Mat compute(cv::Point2d uvLeft, cv::Point2d uvRight);                     // 最小二乘计算坐标
    void    Disptopixelcorr(cv::Mat DispMap);                  // 视差图转为对应的坐标点
    void    allcompute();                       // 计算全部点对并返回
    void    Save_xyz_txt(const char* filename);
    virtual ~computeXYZ();
private:
    vector<cv::Point3d> world_xyz;          // 世界坐标点云
    vector<cv::Point3d> world_rgb;          // 纹理信息
    vector<vector<cv::Point3d>> pointcloud;
    vector<cv::Point2d> pixels_left;        // 像素匹配坐标
    vector<cv::Point2d> pixels_right;       // 像素匹配坐标
    cv::Mat image_left;
    //** 相机内参 **//
    cv::Mat leftIntrinsic;
    cv::Mat leftRotation;
    cv::Mat leftTranslation;
    //**end**//
    cv::Mat rightIntrinsic, rightRotation, rightTranslation;  //  构造时 传递
};

#endif //CUDA_TEST_COMPUTEXYZ_H
