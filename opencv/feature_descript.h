#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/xfeatures2d/cuda.hpp>
#include <opencv2/cudafeatures2d.hpp>
// This file include the census and sift transform class
using namespace std;
class census
{
public:
	census(int id);
	// census 变换
	void census_transform(const cv::Mat input_image, cv::Mat &modified_image, int window_sizex, int window_sizey);
	// census 重载  升级版
    void census_transform(const cv::Mat input_image, cv::Mat &modified_image, int window_sizex, int window_sizey,float threshold_val);
	// 汉明距离计算
	unsigned char census_hanming_dist(long long PL, long long PR);
	// unsigned long census_hanming_dist(long long PL, long long PR);
	~census();
private:
    int id;
	
};
class cost_sift {
public:
	cost_sift(int id);
	void sift_transform(const cv::Mat img0, const cv::Mat img1,
                        vector<vector<vector<float>>>& Desc0,vector<vector<vector<float>>>& Desc1);
	~cost_sift();
private:

};
class WLD{
public:
    WLD(int id);
    void WLD_trans(const cv::Mat input_image, cv::Mat &modified_image,
                    int window_sizex, int window_sizey);
    ~WLD();

private:
};