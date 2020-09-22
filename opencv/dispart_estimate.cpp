#include "dispart_estimate.h"
#define MAX_DISPARITY 164
#define MIN_DISPARITY 1
constexpr auto Invalid_Float=std::numeric_limits<float>::infinity();
extern "C" int cuda_main(float *Cost_disp,float *Cost_Agg,const int Rows,const int Cols, const int D_,
                         const uint *left_image_);   // include cuda_main to aggregation
dispart_estimate::dispart_estimate(const int winsize_x,const int winsize_y) {
    this->winsize_x=winsize_x;  // census transform winsize
    this->winsize_y=winsize_y;
    this->offsetx = (winsize_x - 1) / 2;
    this->offsety = (winsize_y - 1) / 2;
}
dispart_estimate::~dispart_estimate() {
    left.release();
    right.release();
    disp_image.release();
    disp_map.release();
    CensusLeftR.release();
    CensusLeftG.release();
    CensusLeftB.release();
    CensusRightR.release();
    CensusRightG.release();
    CensusRightB.release();
    free(image);
    free(DispLinerImage);
    free(cost_ini);
    free(cost_agg);
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
    this->disp_map=cv::Mat::zeros(left.rows,left.cols,CV_32FC1);            // result disp map
    this->disp_image=cv::Mat::zeros(left.rows,left.cols,CV_8UC1);
    this->image_size.width=left.cols;
    this->image_size.height=left.rows;
    left.copyTo(this->left);
    right.copyTo(this->right);
    cout<<"src image size: "<<disp_map.size()<<endl;
    census *CT_obj_left  = new census(0);
    census *CT_obj_right = new census(1);
    //////////////// split RGB to three invidival threee image mat ///////////////
    ////////////////      census transform for each channel        ///////////////
    cv::Mat left_RGB[3],right_RGB[3];
    cv::split(left ,left_RGB );
    cv::split(right,right_RGB);
    left_RGB[0].copyTo(this->LeftB);
    left_RGB[1].copyTo(this->LeftG);
    left_RGB[2].copyTo(this->LeftR);
    right_RGB[0].copyTo(this->RightB);
    right_RGB[1].copyTo(this->RightG);
    right_RGB[2].copyTo(this->RightR);

    CT_obj_left ->census_transform(this->LeftB,  CensusLeftB, winsize_x,winsize_y);
    CT_obj_left ->census_transform(this->LeftG,  CensusLeftG, winsize_x,winsize_y);
    CT_obj_left ->census_transform(this->LeftR,  CensusLeftR, winsize_x,winsize_y);
    CT_obj_right->census_transform(this->RightB, CensusRightB,winsize_x,winsize_y);
    CT_obj_right->census_transform(this->RightG, CensusRightG,winsize_x,winsize_y);
    CT_obj_right->census_transform(this->RightR, CensusRightR,winsize_x,winsize_y);
    // �ֶ��ͷ��ڴ�
    CT_obj_left ->~census();
    CT_obj_right->~census();
    /////////////////////// end ///////////////////////////////
	// �˺��������Ӳ�ͼ
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
    cv::Mat image_left_gray;
    cv::Mat image_left_roi;
    float weight0=1;
    float weight1=1;
    float weight2=1;
    cvtColor(left,image_left_gray,CV_BGR2GRAY);
    image_left_roi = cv::Mat::zeros(left.rows-2*offsety, right.cols-2*offsetx, CV_8UC1);  // the same size to cost vector's w*h
	vector<vector<vector<float>>> cost_res; // ��ά����  (W-MAX_DISPARITY-offset_x)*(H-2*offsety)*D
    vector<vector<float>> cost_rows;        // ���浱ǰ�е����ص�� D���Ӳ��µĴ���
    vector<float> cost_pix;                 // ���浱ǰ���ص��D���Ӳ��µĴ���
 	for (int i = offsety,image_y=0; i < left.rows-offsety; i++,image_y++) {
		for (int j = offsetx,image_x=0; j < left.cols-offsetx; j++,image_x++) {
		    image_left_roi.at<u_char>(image_y,image_x)=image_left_gray.at<u_char>(i,j);
			for (int m = MIN_DISPARITY; m <=MAX_DISPARITY; m++) {
				int current_right = j-m;
				if(current_right>=offsetx){
                    // ���㲻ͬ�Ӳ��µ� sensus ����
                    float census_hanming_B=(float_t)cost_obj.census_hanming_dist(CensusLeftB.at<float>(i,j),CensusRightB.at<float>(i,current_right));
                    float census_hanming_G=(float_t)cost_obj.census_hanming_dist(CensusLeftG.at<float>(i,j),CensusRightG.at<float>(i,current_right));
                    float census_hanming_R=(float_t)cost_obj.census_hanming_dist(CensusLeftR.at<float>(i,j),CensusRightR.at<float>(i,current_right));
                    float dis_B=abs(CensusLeftB.at<float>(i,j)-CensusRightB.at<float>(i,current_right));
                    float dis_G=abs(CensusLeftB.at<float>(i,j)-CensusRightB.at<float>(i,current_right));
                    float dis_R=abs(CensusLeftB.at<float>(i,j)-CensusRightB.at<float>(i,current_right));
                    float Distence_RGB=(0.11*dis_B+0.59*dis_G+0.3*dis_R)/(255);
                    float Distence_Hanming=(weight0*census_hanming_B+weight1*census_hanming_G+weight2*census_hanming_R)/(3*pow(2,24));
                    // the distance of the target points's vector
                    // float diff=dis_sift(desc0[i][j],desc1[i][current_right]);
                    cost_pix.push_back(Distence_Hanming);   // +diff
				}
				else{
				    cost_pix.push_back(FLT_MAX/2);
				}
			}
			cost_rows.push_back(cost_pix);
			cost_pix.clear();
		}
		cost_res.push_back(cost_rows);
		cost_rows.clear();
	}
 	cout<<"cost_ini mat size: "<<cost_res.size()<<",  "<<cost_res[0].size()<<",  "<<cost_res[0][0].size()<<endl;
    int rows=cost_res.size();     // cost_disp  first index
    int cols=cost_res[0].size();  // second
    int D=cost_res[0][0].size();  // third
    this->rows=rows;
    this->cols=cols;
    this->D=D;
    int index=0;
    int index_image=0;
    DispLinerImage= (float*)malloc(left.cols*left.rows*sizeof(float*));
    image         = (uint*)malloc(rows*cols* sizeof(uint*)); // The same size match the cost_ini and cost_agg;
    cost_ini      = (float*)malloc(rows*cols* D * sizeof(float*));
    cost_agg      = (float*)malloc(rows*cols* D * sizeof(float*));
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            image[index_image++]=image_left_roi.at<uchar>(i,j);
            for(int k=0;k<D;k++){
                cost_ini[index++]=cost_res[i][j][k];
            }
        }
    }
	// ���۾ۺ�
	cuda_main(cost_ini,cost_agg,rows,cols,D,image);     // cuda keral
	// �Ӳ����
    ComputeDisparity();
	// �Ż��Ӳ�
    MedianFilter(DispLinerImage,DispLinerImage,left.cols,left.rows,7);   //  the disp result
    // to convert the shape to w*h
//    for(int i=0;i<left.rows;i++){
//        for(int j=0;j<left.cols;j++){
//            disp_map.data[i*left.cols+j]=DispLinerImage[i*left.cols+j];
//        }
//    }
    // ��ʾ�Ӳ�ͼ
    // ע�⣬������Ʋ�����disp_mat�����ݣ�����������ʾ�ͱ������õġ��������Ҫ�������disparity����������ݣ��������ظ�����
    float min_disp = left.cols, max_disp = 0;
    for (sint32 i = offsety; i < left.rows-offsety; i++) {
        for (sint32 j = offsetx; j < left.cols-offsetx; j++) {
            float disp = DispLinerImage[i * left.cols + j];
            if (disp != Invalid_Float) {
                min_disp = std::min(min_disp, disp);
                max_disp = std::max(max_disp, disp);
            }
        }
    }
    for (sint32 i = offsety; i < left.rows-offsety; i++) {
        for (sint32 j = offsetx; j < left.cols-offsetx; j++) {
            float disp = DispLinerImage[i * left.cols + j];
            if (disp == Invalid_Float) {
                disp_image.at<uchar>(i,j) = 0;
            }
            else {
                disp_image.at<uchar>(i,j) = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
            }
        }
    }
	Disp_Result=this->disp_image;
}
void dispart_estimate::ComputeDisparity() const
{
    const sint32& min_disparity =MIN_DISPARITY;
    const sint32& max_disparity =MAX_DISPARITY;
    const sint32 disp_range = D;
    if(disp_range <= 0) {
        return;
    }
    // ��Ӱ���Ӳ�ͼ
    float* disparity = (float*) malloc(left.cols*left.rows*sizeof(float*));            // liner type
    for(int i=0;i<left.cols*left.rows;i++){
        disparity[i]=0;
    }
    // ��Ӱ��ۺϴ�������
    float* cost_ptr  = (float*)malloc(rows*cols* D * sizeof(float*));

    memcpy(cost_ptr,cost_agg,rows*cols*D*sizeof(float));

    const sint32 width   = this->image_size.width;    // src and the disp
    const sint32 height  = this->image_size.height;
    const bool is_check_unique   = true;
    const float uniqueness_ratio = 0.99;
    // Ϊ�˼ӿ��ȡЧ�ʣ��ѵ������ص����д���ֵ�洢���ֲ�������
    std::vector<float> cost_local(disp_range);
    // ---�����ؼ��������Ӳ�
    for (sint32 i=0; i < rows; i++) {
        for (sint32 j =0; j < cols; j++) {
            float min_cost = FLT_MAX;
            float sec_min_cost = FLT_MAX;
            sint32 best_disparity = 0;
            // ---�����ӲΧ�ڵ����д���ֵ�������С����ֵ����Ӧ���Ӳ�ֵ
            for (sint32 d = min_disparity; d <= max_disparity; d++) {
                const sint32 d_idx = d - min_disparity;
                float cost = cost_local[d_idx] = cost_ptr[i * cols *disp_range + j * disp_range + d_idx];
                if(min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
            }
            if (is_check_unique) {
                // �ٱ���һ�Σ��������С����ֵ
                for (sint32 d = min_disparity; d <= max_disparity; d++) {
                    if (d == best_disparity) {
                        // ������С����ֵ
                        continue;
                    }
                    float cost = cost_local[d - min_disparity];
                    sec_min_cost = std::min(sec_min_cost, cost);
                }
                // �ж�Ψһ��Լ��
                // ��(min-sec)/min < min*(1-uniquness)����Ϊ��Ч����
                if (sec_min_cost - min_cost <= static_cast<float>(min_cost * (1 - uniqueness_ratio))) {
                    disparity[(i+offsety) * width + (j+offsetx)] = Invalid_Float;
                    continue;
                }
            }
            // ---���������
            if (best_disparity == min_disparity || best_disparity == max_disparity) {
                disparity[(i+offsety) * width + (j+offsetx)] = Invalid_Float;
                continue;
            }
            // �����Ӳ�ǰһ���Ӳ�Ĵ���ֵcost_1����һ���Ӳ�Ĵ���ֵcost_2
            const sint32 idx_1 = best_disparity - 1 - min_disparity;
            const sint32 idx_2 = best_disparity + 1 - min_disparity;
            float cost_1 = cost_local[idx_1];
            float cost_2 = cost_local[idx_2];
            // ��һԪ�������߼�ֵ
            float temp= 1;
            float denom = std::max(temp, cost_1 + cost_2 - 2 * min_cost);
            disparity[(i+offsety) * width + (j+offsetx)] = static_cast<float>(best_disparity) + static_cast<float>(cost_1 - cost_2) / (denom * 2.0f);
        }
    }
    memcpy(DispLinerImage,disparity,left.cols*left.rows*sizeof(float));
}