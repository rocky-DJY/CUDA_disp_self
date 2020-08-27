#include "dispart_estimate.h"
#define MAX_DISPARITY 120
#define MIN_DISPARITY 3
extern "C" int cuda_main(vector<vector<vector<float> > > &cost_disp);   // include cuda_main to aggregation
dispart_estimate::dispart_estimate(cv::Mat left, cv::Mat right) {
	this->left  = left;
	this->right = right;
	disp_image=cv::Mat::zeros(left.rows,left.cols,CV_8UC1);
}
// computer the distance of the two vector
float dispart_estimate::dis_sift(const vector<float> Point_desc0, const vector<float> Point_desc1) {
    int len=Point_desc0.size();
    float diff=0;
    for(int i=0;i<len;i++) {
        diff += pow(Point_desc0[i] - Point_desc1[i], 2);
    }
    return pow(diff,0.5);
}
void dispart_estimate::compute_disp(cv::Mat &census_left, cv::Mat &census_right,cv::Mat &Disp_Result) {
	// 此函数计算视差图t
	// 输入  census_left，census_right是经过census变换的图片
	// 创建census对象 代价计算,左图为基准
	// census对象调用hanming距离计算函数 
	// sift 对象 使用对象成员left 和right 计算特征描述子
	// 在代价计算的遍历中  同时加权sift的描述子的cost
	census cost_obj(0);
	// 创建sift对象   代价计算
	cost_sift sift_obj(0);
	vector<vector<vector<float>>> desc0, desc1;
	sift_obj.sift_transform(left, right, desc0, desc1);
    // sift done
    cout<<"sift done"<<endl;
	vector<vector<vector<float>>> cost_res; // 三维矩阵  (W-MAX_DISPARITY)*H*D
 	for (int i = 0; i < left.rows; i++) {
		vector<vector<float>> cost_rows;    // 缓存当前行的像素点的 D个视差下的代价
		for (int j = MAX_DISPARITY; j < left.cols; j++) {
			vector<float> cost_pix;         // 缓存当前像素点的D个视差下的代价
			for (int m = MIN_DISPARITY; m <=MAX_DISPARITY; m++) {
				int current_right = j - m;
				// 计算不同视差下的 sensus 距离
				unsigned char census_hanming=cost_obj.census_hanming_dist(census_left.at<float>(i,j),census_right.at<float>(i,current_right));
				// the distance of the target points's vector
				float diff=dis_sift(desc0[i][j],desc1[i][current_right]);
				//cout<<diff<<", ";
				cost_pix.push_back(census_hanming+diff);
			}
			//cout<<endl;
			cost_rows.push_back(cost_pix);
		}
		cost_res.push_back(cost_rows);
	}
 	cout<<"cost mat size: "<<cost_res.size()<<",  "<<cost_res[0].size()<<",  "<<cost_res[0][0].size()<<endl;
 	//test//
 	for(int i=0;i<20;i++){
 	    for(int j=0;j<20;j++){
 	        for(int k=0;k<20;k++){
 	            cout<<cost_res[i][j][k]<<",";
 	        }
 	        cout<<endl;
 	    }
 	}

	// 代价聚合(选出最优的视差值)
	cuda_main(cost_res); // cuda keral
	// 视差计算
    for(int i=0;i<left.rows;i++){
        for(int j=0;j<left.cols-MAX_DISPARITY;j++){
            vector<float> temp=cost_res[i][j];
            int min_Position=min_element(temp.begin(),temp.end())-temp.begin();
            disp_image.at<uchar>(i,j+MAX_DISPARITY)=min_Position+MIN_DISPARITY;
        }
    }
    cout<<"disp_compute done...and the image size: "<<disp_image.size()<<endl;
	// 优化视差
	// return result
	Disp_Result=disp_image;
}