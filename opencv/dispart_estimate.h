#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include "census.h"
#include <vector>
// this file include the disparity computer
// call the function named computer_disp to return the disprity map
using namespace std;
class dispart_estimate
{
public:
	dispart_estimate(const int winsize_x,const int winsize_y);
	void compute_disp(const cv::Mat src_left,const cv::Mat src_right,cv::Mat &Disp_Result);
	float dis_sift(const vector<float> Point_desc0,const vector<float> Point_desc1 );
private:
	cv::Mat left, right; // source undistort image
	cv::Mat disp_image;  // based on left image
	int winsize_x,winsize_y;
	int offsetx;
	int offsety;
};

