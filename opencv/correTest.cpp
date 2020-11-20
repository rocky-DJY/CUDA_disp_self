//
// Created by maxwell on 9/28/20.
//  this  file to test corre of LR image
#include "dispart_estimate.h"
#include "corre.h"
#define MAX_DISPARITY 233
#define MIN_DISPARITY 10
int trans_val(const float* res,const float* lags,const int dim){
    float res_max=res[0];
    int index=0;
    for(int i=1;i<2*dim-1;i++){
        if(res_max<res[i]){
            res_max=res[i];
            index=i;
        }
    }
    return lags[index];
}
constexpr auto Invalid_Float=std::numeric_limits<float>::infinity();
extern "C" int cuda_main(float *Cost_disp,float *Cost_Agg,const int Rows,const int Cols, const int D_,
                         const uint *left_image_);   // include cuda_main to aggregation
extern "C" void transform(const cv::Mat input_image, cv::Mat &modified_image, int window_sizex, int window_sizey, float threshold_val);
extern "C" void cost_cuda_main(vector<vector<vector<float>>> &cost_nums,
                                double* L_census_R,double* L_census_G,double* L_census_B,
                                double* R_census_R,double* R_census_G,double* R_census_B,
                                cv::Mat left,cv::Mat right);
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
void plot(const float* data,const string win_name,cv::Size size,int scale){
    // data , window name ,data size , scale
    // plot the data row
        cv::Mat L_demo=cv::Mat::zeros(size.height,size.width,CV_8UC1);
        for(int r=0;r<size.width;r++){
            int s=(int)data[r];
            L_demo.at<u_char>(s,r)=255;
        }
        cv::namedWindow(win_name,0);
        cv::resizeWindow(win_name,(int)size.width/scale,(int)size.height/scale);
        cv::imshow(win_name,L_demo);
        cv::waitKey(10);
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
cv::Mat dispart_estimate::compute_disp(const cv::Mat left, const cv::Mat right,cv::Mat &Disp_Result) {
    // inout src left and right image output dispimage
    // census transform
    // new census obj 需要手动释放对象的内存
    this->disp_map=cv::Mat::zeros(left.rows,left.cols,CV_32FC1);            // result disp map
    this->disp_image=cv::Mat::zeros(left.rows,left.cols,CV_8UC1);
    this->image_size.width=left.cols;
    this->image_size.height=left.rows;
    this->rows=left.rows;
    this->cols=left.cols;
    left.copyTo(this->left);
    right.copyTo(this->right);
    cv::Mat left_RGB[3],right_RGB[3];
    cv::split(left ,left_RGB );
    cv::split(right,right_RGB);
    /////////////////////// end ///////////////////////////////
    // 此函数计算视差图
    // 输入  census_left，census_right是经过census变换的图片
    // 创建census对象 代价计算,左图为基准
    // census对象调用hanming距离计算函数
    cv::Mat image_left_gray;
    cv::Mat image_left_roi;
    cvtColor(left,image_left_gray,CV_BGR2GRAY);
    image_left_roi = cv::Mat::zeros(left.rows-2*offsety, right.cols-2*offsetx, CV_8UC1);  // the same size to cost vector's w*h
    int m_dim=left.rows-2*offsety;
    int n_dim=left.cols-2*offsetx;
    int d_dim=MAX_DISPARITY-MIN_DISPARITY+1;
    // 三维矩阵  (H-2*offsety)*(W-2*offset_x)*D
    vector<vector<vector<float>>> cost_res(m_dim,vector<vector<float>>(n_dim,vector<float>(d_dim,0)));
//    vector<vector<float>> cost_rows;          // 缓存当前行的像素点的 D个视差下的代价
//    vector<float> cost_pix;                   // 缓存当前像素点的D个视差下的代价
//    cost_pix.resize(MAX_DISPARITY-MIN_DISPARITY+1);
    float *L_0 =new float[left.cols];       // 0:B 1:G 2:R
    float *R_0 =new float[left.cols];
    float *L_1 =new float[left.cols];
    float *R_1 =new float[left.cols];
    float *L_2 =new float[left.cols];
    float *R_2 =new float[left.cols];
    double* left_census_R= (double *)malloc(sizeof(double) * rows * cols);
    double* left_census_G= (double *)malloc(sizeof(double) * rows * cols);
    double* left_census_B= (double *)malloc(sizeof(double) * rows * cols);
    double* right_census_R= (double *)malloc(sizeof(double) * rows * cols);
    double* right_census_G= (double *)malloc(sizeof(double) * rows * cols);
    double* right_census_B= (double *)malloc(sizeof(double) * rows * cols);
    //////////////** census transform **////////////////
//    census left_census(0);
//    census right_census(1);
//    float threshold_val=5.6;  // 1.8
//    auto start0=system_clock::now();
//    left_census.census_transform(left_RGB[0],CensusLeftB,winsize_x,winsize_y,threshold_val);
//    left_census.census_transform(left_RGB[1],CensusLeftG,winsize_x,winsize_y,threshold_val);
//    left_census.census_transform(left_RGB[2],CensusLeftR,winsize_x,winsize_y,threshold_val);
//    right_census.census_transform(right_RGB[0],CensusRightB,winsize_x,winsize_y,threshold_val);
//    right_census.census_transform(right_RGB[1],CensusRightG,winsize_x,winsize_y,threshold_val);
//    right_census.census_transform(right_RGB[2],CensusRightR,winsize_x,winsize_y,threshold_val);
//    auto last0=system_clock::now();
//    auto duration0=duration_cast<microseconds>(last0-start0);
//    cout<<"census_cpu total time: "<<double(duration0.count())*microseconds::period::num/microseconds::period::den<<endl;
    /////////////cuda census//////////////
    float threshold_val=1.8;  // 1.8
    auto start=system_clock::now();
    transform(left_RGB[0],CensusLeftB,winsize_x,winsize_y,threshold_val);
    transform(left_RGB[1],CensusLeftG,winsize_x,winsize_y,threshold_val);
    transform(left_RGB[2],CensusLeftR,winsize_x,winsize_y,threshold_val);
    transform(right_RGB[0],CensusRightB,winsize_x,winsize_y,threshold_val);
    transform(right_RGB[1],CensusRightG,winsize_x,winsize_y,threshold_val);
    transform(right_RGB[2],CensusRightR,winsize_x,winsize_y,threshold_val);
    auto last=system_clock::now();
    auto duration=duration_cast<microseconds>(last-start);
    cout<<"census_cuda total time: "<<double(duration.count())*microseconds::period::num/microseconds::period::den<<endl;
    //////////////** end **//////////////////
    ///////*****data flat****////////////////
    int _index=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            left_census_B[_index]=CensusLeftB.at<double>(i,j);
            left_census_G[_index]=CensusLeftG.at<double>(i,j);
            left_census_R[_index]=CensusLeftR.at<double>(i,j);
            right_census_B[_index]=CensusRightB.at<double>(i,j);
            right_census_G[_index]=CensusRightG.at<double>(i,j);
            right_census_R[_index]=CensusRightR.at<double>(i,j);
            _index++;
        }
    }
    ///////*****end*****/////////////////////
    //////////////**********************/////////////////////////////
    int len=11;   //odd number must
    //////////////**********************/////////////////////////////
    float *temp_L_0=new float[len];
    float *temp_R_0=new float[len];
    float *temp_L_1=new float[len];
    float *temp_R_1=new float[len];
    float *temp_L_2=new float[len];
    float *temp_R_2=new float[len];
    float *CO_weight =new float[3];
    float *CT_weight =new float[3];
    float uni_size=255;
    vector<float> nums1;
    vector<float> nums2;
    int image_y,image_x;  // the ROI point index
    ////***  cuda   cost compute ****//////////
    auto _start=system_clock::now();
    cost_cuda_main(cost_res,left_census_R,left_census_G,left_census_B,right_census_R,right_census_G,right_census_B,this->left,this->right);
    auto _last=system_clock::now();
    auto _duration=duration_cast<microseconds>(_last-_start);
    cout<<"cost_cuda total time: "<<double(_duration.count())*microseconds::period::num/microseconds::period::den<<endl;
    ////////**** end ****//////////////
    for (int i = this->offsety; i < left.rows-offsety; i++) { // row index
        image_y=i-offsety;
        for (int j = offsetx; j < left.cols-offsetx; j++) {  // col index
            image_x=j-offsetx;
            image_left_roi.at<u_char>(image_y,image_x)=image_left_gray.at<u_char>(i,j);
        }
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
    // 代价聚合
    cuda_main(cost_ini,cost_agg,rows,cols,D,image);     // cuda keral
    // 视差计算
    ComputeDisparity();
    // 优化视差
    MedianFilter(DispLinerImage,DispLinerImage,left.cols,left.rows,7);   //  the disp result
    // 显示视差图
    // 注意，计算点云不能用disp_mat的数据，它是用来显示和保存结果用的。计算点云要用上面的disparity数组里的数据，是子像素浮点数
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
                disp_map.at<float>(i,j)=0;
            }
            else {
                disp_image.at<uchar>(i,j) = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
                disp_map.at<float>(i,j)=disp;
            }
        }
    }
    //// release memory  /////
    delete temp_L_0;
    delete temp_L_1;
    delete temp_L_2;
    delete temp_R_0;
    delete temp_R_1;
    delete temp_R_2;
    delete CT_weight;
    delete CO_weight;
    delete L_0;
    delete L_1;
    delete L_2;
    delete R_0;
    delete R_1;
    delete R_2;
    free(left_census_B);
    free(left_census_G);
    free(left_census_R);
    free(right_census_B);
    free(right_census_G);
    free(right_census_R);
    //////  end ///////
    Disp_Result=this->disp_image;
    return disp_map;
}
void dispart_estimate::ComputeDisparity() const
{
    const sint32& min_disparity =MIN_DISPARITY;
    const sint32& max_disparity =MAX_DISPARITY;
    const sint32 disp_range = D;
    if(disp_range <= 0) {
        return;
    }
    // 左影像视差图
    float* disparity = (float*) malloc(left.cols*left.rows*sizeof(float*));            // liner type
    for(int i=0;i<left.cols*left.rows;i++){
        disparity[i]=0;
    }
    // 左影像聚合代价数组
    float* cost_ptr  = (float*)malloc(rows*cols* D * sizeof(float*));

    memcpy(cost_ptr,cost_agg,rows*cols*D*sizeof(float));   // chose the cost mat

    const sint32 width   = this->image_size.width;         // src and the disp
    const sint32 height  = this->image_size.height;
    const bool is_check_unique   = true;
    const float uniqueness_ratio = 0.99;
    // 为了加快读取效率，把单个像素的所有代价值存储到局部数组里
    std::vector<float> cost_local(disp_range);
    // ---逐像素计算最优视差
    for (sint32 i=0; i < rows; i++) {
        for (sint32 j =0; j < cols; j++) {
            float min_cost = FLT_MAX;
            float sec_min_cost = FLT_MAX;
            sint32 best_disparity = 0;
            // ---遍历视差范围内的所有代价值，输出最小代价值及对应的视差值
            for (sint32 d = min_disparity; d <= max_disparity; d++) {
                const sint32 d_idx = d - min_disparity;
                float cost = cost_local[d_idx] = cost_ptr[i * cols *disp_range + j * disp_range + d_idx];
                if(min_cost > cost) {
                    min_cost = cost;
                    best_disparity = d;
                }
            }
            if (is_check_unique) {
                // 再遍历一次，输出次最小代价值
                for (sint32 d = min_disparity; d <= max_disparity; d++) {
                    if (d == best_disparity) {
                        // 跳过最小代价值
                        continue;
                    }
                    float cost = cost_local[d - min_disparity];
                    sec_min_cost = std::min(sec_min_cost, cost);
                }
                // 判断唯一性约束
                // 若(min-sec)/min < min*(1-uniquness)，则为无效估计
                if (sec_min_cost - min_cost <= static_cast<float>(min_cost * (1 - uniqueness_ratio))) {
                    disparity[(i+offsety) * width + (j+offsetx)] = Invalid_Float;
                    continue;
                }
            }
            // ---子像素拟合
            if (best_disparity == min_disparity || best_disparity == max_disparity) {
                disparity[(i+offsety) * width + (j+offsetx)] = Invalid_Float;
                continue;
            }
            // 最优视差前一个视差的代价值cost_1，后一个视差的代价值cost_2
            const sint32 idx_1 = best_disparity - 1 - min_disparity;
            const sint32 idx_2 = best_disparity + 1 - min_disparity;
            float cost_1 = cost_local[idx_1];
            float cost_2 = cost_local[idx_2];
            // 解一元二次曲线极值
            float temp= 1;
            float denom = std::max(temp, cost_1 + cost_2 - 2 * min_cost);
            disparity[(i+offsety) * width + (j+offsetx)] = static_cast<float>(best_disparity) + static_cast<float>(cost_1 - cost_2) / (denom * 2.0f);
        }
    }
    memcpy(DispLinerImage,disparity,left.cols*left.rows*sizeof(float));
}
