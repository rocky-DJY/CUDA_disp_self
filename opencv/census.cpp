#include "census.h"
#include <tiff.h>
census::census(int id) {
	//cout << "creat_census_id: "<<id << endl;
}
census::~census() {
	//cout << "delete census_obj" << endl;
}
void census::census_transform(const cv::Mat input_image, cv::Mat &modified_image, int window_sizex, int window_sizey) {
		int image_height = input_image.rows;
		int image_width = input_image.cols;
		cv::Mat image_src;
		cvtColor(input_image,image_src,CV_BGR2GRAY);
		modified_image = cv::Mat::zeros(image_height, image_width, CV_32FC1);  // 64   census result
		//-----------   census变换     -------------//
		int offsetx = (window_sizex - 1) / 2;
		int offsety = (window_sizey - 1) / 2;
		for (int j = 0; j < image_width - window_sizex; j++){       //col
			for (int i = 0; i < image_height - window_sizey; i++){  //row
				uint32 census = 0u;    // unsigned long
				uint8_t current_pixel = image_src.at<uint8_t>(i + offsety, j + offsetx); //窗口中心像素  uchar
				cv::Rect roi(j, i, window_sizex, window_sizey); //方形窗口
				cv::Mat window(image_src, roi);
				for (int a = 0; a < window_sizey; a++){
					for (int b = 0; b < window_sizex; b++){
						if (!(a == offsety && b == offsetx)){    //中心像素不做判断
							census = census << 1;                //左移1位
						    uint8_t temp_value = window.at<uint8_t>(a, b);
						    if (temp_value < current_pixel){    //当前像素小于中心像素 01
							    census += 1;
						    }
					    }
				    }
				}
				modified_image.at<float_t>(i + offsety, j + offsetx) = census;
			}
		}
}
unsigned char census::census_hanming_dist(long long PL, long long PR) {
	// Fast Hamming distance algorithm
	// number 大于0  代表不同两个census序列互相转换的次数
	unsigned char number = 0;
	long long v;
	v = PL ^ PR;            /* ^ 异或运算 不同为1 相同为0*/
	while (v)
	{
		v &= (v - 1);            /* & 与运算*/
		number++;
	}
	return number;
}
cost_sift::cost_sift(int id) {
    // constructor
}
cost_sift::~cost_sift() {
	//delete the class's variable member
}
void cost_sift::sift_transform(const cv::Mat img0,const cv::Mat img1,
                               vector<vector<vector<float>>>& Desc0,vector<vector<vector<float>>>& Desc1){
	cv::cuda::GpuMat left_img;
	cv::cuda::GpuMat right_img;
	cv::Mat img0_gray, img1_gray;
	cvtColor(img0, img0_gray, CV_BGR2GRAY);
	cvtColor(img1, img1_gray, CV_BGR2GRAY);
	left_img.upload(img0_gray);
	right_img.upload(img1_gray);

	//下面这个函数的原型是：
	//explicit SURF_CUDA(
	//double _hessianThreshold,    //SURF海森特征点阈值 越小检测的特征点越多
	//int _nOctaves=4,             //尺度金字塔个数
	//int _nOctaveLayers=2,		   //每一个尺度金字塔层数
	//bool _extended=false,		   //如果true那么得到的描述子是128维，否则是64维
	//float _keypointsRatio=0.01f,
	//bool _upright = false
	//);
	//要理解这几个参数涉及SURF的原理
	int dim = 64;
	cv::cuda::SURF_CUDA surf(50, 4, 3);
	/*分配下面几个GpuMat存储keypoint和相应的descriptor*/
	vector<cv::KeyPoint> key0(img0.rows*img0.cols);
	vector<cv::KeyPoint> key1(img0.rows*img0.cols);
	vector<cv::KeyPoint> key_demo;
	cv::cuda::GpuMat keypt0, keypt1;
	cv::cuda::GpuMat desc0,  desc1;
	int point = 0;
	// 自定义特征点  即每个像素点  平行换行遍历方式 后续描述子的排列以此来决定
	for (int i = 0; i < img0.rows; i++)
		for (int j = 0; j < img0.cols; j++) {
			key0[point].pt.x = j;
			key0[point].pt.y = i;
			key0[point].size = 60; // base 30
			key0[point].octave = 0;
			key1[point].pt.x = j;
			key1[point].pt.y = i;
			key1[point].size = 60;
			key1[point].octave = 0;
			point++;
		}
	surf.uploadKeypoints(key0, keypt0);
	surf.uploadKeypoints(key1, keypt1);
	//keypt0.upload(key0);
	//keypt1.upload(key1);
	//描述子生成   gpu_mat (i,j)=(w,h)与普通mat相反
    vector<float> descriptor0,descriptor1;
	surf(left_img,  cv::cuda::GpuMat(), keypt0, desc0,true); //使用自定义的keypt
	surf(right_img, cv::cuda::GpuMat(), keypt1, desc1,true); 
	surf.downloadDescriptors(desc0, descriptor0);  // 下载描述子给vector
	surf.downloadDescriptors(desc1, descriptor1);
	// cout << keypt0.size() << endl;
	// 将64维向量按照w*h的分布储存
	int index = 0;
	int counter0=0;
	int counter1=0;
	vector<vector<vector<float>>> desc0_wh;   // w*h分布的点的向量
	vector<vector<vector<float>>> desc1_wh;   // w*h分布的点的向量
	for (int i = 0; i < img0_gray.rows; i++) { 
		vector<vector<float>> desc0_point_row;  // 每一行的每一个点的向量
		vector<vector<float>> desc1_point_row;
		for (int j = 0; j < img0_gray.cols; j++) {
			vector<float> desc0_point;          // 每64个为一个点的向量
			vector<float> desc1_point;
			for (int m = 0; m <dim ; m++) {
				desc0_point.push_back(descriptor0[index]);
				desc1_point.push_back(descriptor0[index]);
                if(isnan(descriptor0[index]))
                    counter0++;
                if(isnan(descriptor1[index]))
                    counter1++;
				index++;
			}
			desc0_point_row.push_back(desc0_point);
			desc1_point_row.push_back(desc1_point);
		}
		desc0_wh.push_back(desc0_point_row);
		desc1_wh.push_back(desc1_point_row);
	}
	cout<<"counter0: "<<counter0<<endl;
	cout<<"counter1: "<<counter1<<endl;
	Desc0=desc0_wh;
	Desc1=desc1_wh;
	//cout << keypt0.size().width << endl;
	//cout << desc0.size().width << endl;
	//cout << "descriptor generate" << endl;
}