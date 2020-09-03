#include "dispart_estimate.h"
#define MAX_DISPARITY 132
#define MIN_DISPARITY 1
extern "C" int cuda_main(vector<vector<vector<float> > > &cost_disp);   // include cuda_main to aggregation
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
    // new census obj ��Ҫ�ֶ��ͷŶ�����ڴ�
    this->disp_image=cv::Mat::zeros(left.rows,left.cols,CV_8UC1);   // result disp image
    cout<<"src image size: "<<disp_image.size()<<endl;
    census *CT_obj_left  = new census(0);
    census *CT_obj_right = new census(1);
    cv::Mat census_image_left, census_image_right; // census �任��� ʹ��64F datatype
    CT_obj_left ->census_transform(left,  census_image_left, winsize_x,winsize_y);
    CT_obj_right->census_transform(right, census_image_right,winsize_x,winsize_y);
    //cv::imshow("census_transfrom_L", census_image_left);
    //cv::imshow("census_transfrom_R", census_image_right);
    // �ֶ��ͷ��ڴ�
    CT_obj_left ->~census();
    CT_obj_right->~census();
	// �˺��������Ӳ�ͼt
	// ����  census_left��census_right�Ǿ���census�任��ͼƬ
	// ����census���� ���ۼ���,��ͼΪ��׼
	// census�������hanming������㺯�� 
	// sift ���� ʹ�ö����Աleft ��right ��������������
	// �ڴ��ۼ���ı�����  ͬʱ��Ȩsift�������ӵ�cost
	census cost_obj(0);
	// ����sift����   ���ۼ���
	cost_sift sift_obj(0);
	vector<vector<vector<float>>> desc0, desc1;
	sift_obj.sift_transform(left, right, desc0, desc1);
    // sift done
    cout<<"sift done"<<endl;
	vector<vector<vector<float>>> cost_res; // ��ά����  (W-MAX_DISPARITY-winsize_x)*(H-winsize_y)*D
    vector<vector<float>> cost_rows;        // ���浱ǰ�е����ص�� D���Ӳ��µĴ���
    vector<float> cost_pix;                 // ���浱ǰ���ص��D���Ӳ��µĴ���
 	for (int i = offsety; i < left.rows-offsety; i++) {
		for (int j = MAX_DISPARITY; j < left.cols-offsetx; j++) {
			for (int m = MIN_DISPARITY; m <=MAX_DISPARITY; m++) {
				int current_right = j - m;
				// ���㲻ͬ�Ӳ��µ� sensus ����
				float census_hanming=(float_t)cost_obj.census_hanming_dist(census_image_left.at<float>(i,j),census_image_right.at<float>(i,current_right));
				// the distance of the target points's vector
				float diff=dis_sift(desc0[i][j],desc1[i][current_right]);
				// cout<<diff<<", ";
//				printf("%f--",census_image_left.at<float>(i,j));
//				printf("%f-->",census_image_right.at<float>(i,current_right));
//				printf("%f",census_hanming);
//				printf("\n");
				cost_pix.push_back(census_hanming);   // +diff
			}
			//cout<<endl;
			cost_rows.push_back(cost_pix);
			cost_pix.clear();
		}
		cost_res.push_back(cost_rows);
		cost_rows.clear();
	}
 	cout<<"cost mat size: "<<cost_res.size()<<",  "<<cost_res[0].size()<<",  "<<cost_res[0][0].size()<<endl;
	// ���۾ۺ�(ѡ�����ŵ��Ӳ�ֵ)
	cuda_main(cost_res); // cuda keral

	// �Ӳ����
    for(int i=0;i<left.rows-winsize_y;i++){
        for(int j=0;j<left.cols-MAX_DISPARITY-offsetx;j++){
            vector<float> temp=cost_res[i][j];
            int min_Position=min_element(temp.begin(),temp.end())-temp.begin();
            //disp_image.at<uchar>(i,j+MAX_DISPARITY)=100+(min_Position+MIN_DISPARITY)*255/120;
            this->disp_image.at<u_char>(i+offsety,j+MAX_DISPARITY)=3*static_cast<uchar>(min_Position+MIN_DISPARITY);
            //this->disp_image.at<u_char>(i+offsety,j+MAX_DISPARITY)= static_cast<uchar>(min_Position)/(MAX_DISPARITY-MIN_DISPARITY)*255;
        }
    }
    cout<<"disp_compute done...and the image size: "<<disp_image.size()<<endl;
	// �Ż��Ӳ�
	// return result
	Disp_Result=this->disp_image;
}