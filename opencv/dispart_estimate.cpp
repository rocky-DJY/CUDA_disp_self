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
	vector<vector<vector<float>>> cost_res; // ��ά����  (W-MAX_DISPARITY)*H*D
 	for (int i = 0; i < left.rows; i++) {
		vector<vector<float>> cost_rows;    // ���浱ǰ�е����ص�� D���Ӳ��µĴ���
		for (int j = MAX_DISPARITY; j < left.cols; j++) {
			vector<float> cost_pix;         // ���浱ǰ���ص��D���Ӳ��µĴ���
			for (int m = MIN_DISPARITY; m <=MAX_DISPARITY; m++) {
				int current_right = j - m;
				// ���㲻ͬ�Ӳ��µ� sensus ����
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

	// ���۾ۺ�(ѡ�����ŵ��Ӳ�ֵ)
	cuda_main(cost_res); // cuda keral
	// �Ӳ����
    for(int i=0;i<left.rows;i++){
        for(int j=0;j<left.cols-MAX_DISPARITY;j++){
            vector<float> temp=cost_res[i][j];
            int min_Position=min_element(temp.begin(),temp.end())-temp.begin();
            disp_image.at<uchar>(i,j+MAX_DISPARITY)=min_Position+MIN_DISPARITY;
        }
    }
    cout<<"disp_compute done...and the image size: "<<disp_image.size()<<endl;
	// �Ż��Ӳ�
	// return result
	Disp_Result=disp_image;
}