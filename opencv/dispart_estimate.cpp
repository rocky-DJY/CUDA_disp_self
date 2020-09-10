#include "dispart_estimate.h"
#define MAX_DISPARITY 164
#define MIN_DISPARITY 1
extern "C" int cuda_main(vector<vector<vector<float> > > &cost_disp,const cv::Mat left_image);   // include cuda_main to aggregation
dispart_estimate::dispart_estimate(const int winsize_x,const int winsize_y) {
    this->winsize_x=winsize_x;
    this->winsize_y=winsize_y;
    this->offsetx = (winsize_x - 1) / 2;
    this->offsety = (winsize_y - 1) / 2;
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
void dispart_estimate::compute_disp(const cv::Mat left, const cv::Mat right,cv::Mat &Disp_Result) {
    // inout src left and right image   output dispimage
    // census transform
    // new census obj 需要手动释放对象的内存
    this->disp_map=cv::Mat::zeros(left.rows,left.cols,CV_8UC1);            // result disp map
    this->disp_image=cv::Mat::zeros(left.rows,left.cols,CV_8UC1);
    cout<<"src image size: "<<disp_map.size()<<endl;
    census *CT_obj_left  = new census(0);
    census *CT_obj_right = new census(1);
    cv::Mat census_image_left, census_image_right; // census 变换结果 使用64F datatype
    CT_obj_left ->census_transform(left,  census_image_left, winsize_x,winsize_y);
    CT_obj_right->census_transform(right, census_image_right,winsize_x,winsize_y);
    //cv::imshow("census_transfrom_L", census_image_left);
    //cv::imshow("census_transfrom_R", census_image_right);
    // 手动释放内存
    CT_obj_left ->~census();
    CT_obj_right->~census();
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
    cv::Mat image_left_gray;
    cv::Mat image_left_roi;
    cvtColor(left,image_left_gray,CV_BGR2GRAY);
    image_left_roi = cv::Mat::zeros(left.rows-2*offsety, right.cols-MAX_DISPARITY-offsetx, CV_8UC1);  // the same size to cost vector's w*h
	vector<vector<vector<float>>> cost_res; // 三维矩阵  (W-MAX_DISPARITY-offset_x)*(H-2*offsety)*D
    vector<vector<float>> cost_rows;        // 缓存当前行的像素点的 D个视差下的代价
    vector<float> cost_pix;                 // 缓存当前像素点的D个视差下的代价
 	for (int i = offsety,image_y=0; i < left.rows-offsety; i++,image_y++) {
		for (int j = MAX_DISPARITY,image_x=0; j < left.cols-offsetx; j++,image_x++) {
		    image_left_roi.at<u_char>(image_y,image_x)=image_left_gray.at<u_char>(i,j);
			for (int m = MIN_DISPARITY; m <=MAX_DISPARITY; m++) {
				int current_right = j - m;
				// 计算不同视差下的 sensus 距离
				float census_hanming=(float_t)cost_obj.census_hanming_dist(census_image_left.at<float>(i,j),census_image_right.at<float>(i,current_right));
				// the distance of the target points's vector
				float diff=dis_sift(desc0[i][j],desc1[i][current_right]);
//				// cout<<diff<<", ";
//				printf("%f--",census_image_left.at<float>(i,j));
//				printf("%f-->",census_image_right.at<float>(i,current_right));
//				printf("%f",census_hanming);
//				printf("\n");
				cost_pix.push_back(census_hanming+diff*0.01);   // +diff
			}
			//cout<<endl;
			cost_rows.push_back(cost_pix);
			cost_pix.clear();
		}
		cost_res.push_back(cost_rows);
		cost_rows.clear();
	}
 	cout<<"cost mat size: "<<cost_res.size()<<",  "<<cost_res[0].size()<<",  "<<cost_res[0][0].size()<<endl;
    // roi stract
	// 代价聚合(选出最优的视差值)
	cuda_main(cost_res,image_left_roi);     // cuda keral
	// 视差计算
	uint max_disp=0;
	uint min_disp=0;
    for(int i=0;i<left.rows-winsize_y;i++){
        for(int j=0;j<left.cols-MAX_DISPARITY-offsetx;j++){
            vector<float> temp=cost_res[i][j];
            int min_Position=min_element(temp.begin(),temp.end())-temp.begin();
            uint disp=min_Position+MIN_DISPARITY;
            max_disp=max(max_disp,disp);
            disp_map.at<u_char>(i+offsety,j+MAX_DISPARITY)=disp;
        }
    }
    // cout<<"here:"<<endl;
    for(int i=0;i<disp_image.rows;i++){
        for(int j=0;j<disp_image.cols;j++){
            if(disp_map.at<u_char>(i,j)!=0){
                uint current=disp_map.at<u_char>(i,j);
                disp_image.at<u_char>(i,j)= static_cast<u_char>(255*(current-min_disp)/(max_disp-min_disp));
            }
        }
    }
    cout<<"disp_compute done...and the image size: "<<disp_map.size()<<endl;
	// 优化视差
	// return result
    MedianFilter(disp_image,disp_image,cv::Size(9,9));
	Disp_Result=this->disp_image;
}