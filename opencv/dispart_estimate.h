#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "feature_descript.h"
#include <vector>
#include "image_process_utl.h"
#include <chrono>
typedef int32_t sint32;
// this file include the disparity computer
// call the function named computer_disp to return the disprity map
using namespace std;
using namespace chrono;
class dispart_estimate{
public:
	dispart_estimate(const int winsize_x,const int winsize_y);
	cv::Mat compute_disp(const cv::Mat src_left,const cv::Mat src_right,cv::Mat &Disp_Result);
	float dis_sift(const vector<float> Point_desc0,const vector<float> Point_desc1 );
    void ComputeDisparity() const;
    ~dispart_estimate();
private:
	cv::Mat left, right; // source undistort image
	cv::Mat disp_map;    // based on left image
	cv::Mat disp_image;  // to show
	cv::Size image_size;
    cv::Mat CensusLeftB;
    cv::Mat CensusLeftG;
    cv::Mat CensusLeftR;
    cv::Mat CensusRightB;
    cv::Mat CensusRightG;
    cv::Mat CensusRightR;
    cv::Mat LeftB;
    cv::Mat LeftG;
    cv::Mat LeftR;
    cv::Mat RightB;
    cv::Mat RightG;
    cv::Mat RightR;
    uint  *image;
    float *DispLinerImage;
    float *cost_ini;
    float *cost_agg;
    int rows,cols,D;
    int winsize_x,winsize_y;
    int offsetx;
    int offsety;
};

