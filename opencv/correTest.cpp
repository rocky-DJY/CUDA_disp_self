//
// Created by maxwell on 9/28/20.
//  this  file to test corre of LR image
#include "dispart_estimate.h"
#include "corre.h"
#define MAX_DISPARITY 168
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
void dispart_estimate::compute_disp(const cv::Mat left, const cv::Mat right,cv::Mat &Disp_Result) {
    // inout src left and right image   output dispimage
    // census transform
    // new census obj 需要手动释放对象的内存
    this->disp_map=cv::Mat::zeros(left.rows,left.cols,CV_32FC1);            // result disp map
    this->disp_image=cv::Mat::zeros(left.rows,left.cols,CV_8UC1);
    this->image_size.width=left.cols;
    this->image_size.height=left.rows;
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
    vector<vector<vector<float>>> cost_res; // 三维矩阵  (W-MAX_DISPARITY-offset_x)*(H-2*offsety)*D
    vector<vector<float>> cost_rows;        // 缓存当前行的像素点的 D个视差下的代价
    vector<float> cost_pix;                 // 缓存当前像素点的D个视差下的代价
    float *L_0 =new float[left.cols];       // 0:B 1:G 2:R
    float *R_0 =new float[left.cols];
    float *L_1 =new float[left.cols];
    float *R_1 =new float[left.cols];
    float *L_2 =new float[left.cols];
    float *R_2 =new float[left.cols];
    //////////////** census transform **//////////////////
    census left_census(0);
    census right_census(1);
    float threshold_val=1.8;
    left_census.census_transform(left_RGB[0],CensusLeftB,winsize_x,winsize_y,threshold_val);
    left_census.census_transform(left_RGB[1],CensusLeftG,winsize_x,winsize_y,threshold_val);
    left_census.census_transform(left_RGB[2],CensusLeftR,winsize_x,winsize_y,threshold_val);
    right_census.census_transform(right_RGB[0],CensusRightB,winsize_x,winsize_y,threshold_val);
    right_census.census_transform(right_RGB[1],CensusRightG,winsize_x,winsize_y,threshold_val);
    right_census.census_transform(right_RGB[2],CensusRightR,winsize_x,winsize_y,threshold_val);
    //////////////** end **//////////////////
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
    for (int i = offsety,image_y=0; i < left.rows-offsety; i++,image_y++) { // row index
        // first get the one row image data
        for(int T=0;T<left.cols;T++){
            L_0[T]= (float)left_RGB[0].at<u_char>(i,T)/uni_size;
            R_0[T]= (float)right_RGB[0].at<u_char>(i,T)/uni_size;
            L_1[T]= (float)left_RGB[1].at<u_char>(i,T)/uni_size;
            R_1[T]= (float)right_RGB[1].at<u_char>(i,T)/uni_size;
            L_2[T]= (float)left_RGB[2].at<u_char>(i,T)/uni_size;
            R_2[T]= (float)right_RGB[2].at<u_char>(i,T)/uni_size;
        }
        //plot(L,"Ldata",cv::Size(1920,256),4);
        //plot(R,"Rdata",cv::Size(1920,256),4);
        for (int j = offsetx,image_x=0; j < left.cols-offsetx; j++,image_x++) {  // col index
            image_left_roi.at<u_char>(image_y,image_x)=image_left_gray.at<u_char>(i,j);
            //  get the left block data
            for(int front_index=j,back_index=j,front=(int)len/2,back=(int)len/2;back_index>=j-len/2;front_index++,back_index--){
                if(front_index>left.cols-1){
                    temp_L_0[front]=0.0000f;
                    temp_L_1[front]=0.0000f;
                    temp_L_2[front]=0.0000f;
                    front++;
                }
                else {
                    temp_L_0[front]=L_0[front_index];
                    temp_L_1[front]=L_1[front_index];
                    temp_L_2[front]=L_2[front_index];
                    front++;
                }
                if(back_index<0){
                    temp_L_0[back]=0.0000f;
                    temp_L_1[back]=0.0000f;
                    temp_L_2[back]=0.0000f;
                    back--;
                }
                else {
                    temp_L_0[back] = L_0[back_index];
                    temp_L_1[back] = L_1[back_index];
                    temp_L_2[back] = L_2[back_index];
                    back--;
                }
            }
            // plot(temp_L,"L_temp",cv::Size(len,256),1);
            for (int m = MIN_DISPARITY; m <=MAX_DISPARITY; m++) {
                int current_right = j-m;
                if(current_right>=offsetx){
                    // get the right block data
                    for(int front_index=current_right,back_index=current_right,front=(int)len/2,back=(int)len/2;
                        back_index>=current_right-len/2;front_index++,back_index--){
                        if(front_index>left.cols-1){
                            temp_R_0[front]=0.0000f;
                            temp_R_1[front]=0.0000f;
                            temp_R_2[front]=0.0000f;
                            front++;
                        }
                        else {
                            temp_R_0[front]=R_0[front_index];
                            temp_R_1[front]=R_1[front_index];
                            temp_R_2[front]=R_2[front_index];
                            front++;
                        }
                        if(back_index<0){
                            temp_R_0[back]=0.0000f;
                            temp_R_1[back]=0.0000f;
                            temp_R_2[back]=0.0000f;
                            back--;
                        }
                        else{
                            temp_R_0[back]=R_0[back_index];
                            temp_R_1[back]=R_1[back_index];
                            temp_R_2[back]=R_2[back_index];
                            back--;
                        }
                        // cout<<"front: "<<front<<" back: "<<back<<"   len: "<<len<<endl;
                    }
                    // plot(temp_R,"R_temp",cv::Size(len,256),1);
                    // corre(temp_L,temp_R,res,lags,len);
                    // int curr_trans_val=trans_val(res,lags,len);
                    // the disstence of the  two arrays trans val and assuming trans;
                    float diss0=abs(calculate_corss_correlation(temp_L_0,temp_R_0,len));  // 数值越大表示代价越小
                    float diss1=abs(calculate_corss_correlation(temp_L_1,temp_R_1,len));
                    float diss2=abs(calculate_corss_correlation(temp_L_2,temp_R_2,len));
                    float aver_corr=(diss0+diss1+diss2)/3;
                    float sum_abs_co=abs(diss0-aver_corr)+abs(diss1-aver_corr)+abs(diss2-aver_corr);
                    if(sum_abs_co!=0){
                        CO_weight[0]=abs(diss0-aver_corr)/sum_abs_co;
                        CO_weight[1]=abs(diss1-aver_corr)/sum_abs_co;
                        CO_weight[2]=abs(diss2-aver_corr)/sum_abs_co;
                    }
                    else{
                        CO_weight[0]=0.3;
                        CO_weight[1]=0.3;
                        CO_weight[2]=0.3;
                    }
                    // 计算不同视差下的 sensus 距离
                    //float hi=(float_t)left_census.census_hanming_dist(CensusLeftB.at<double>(i,j),CensusRightB.at<double>(i,current_right));
                    float census_hanming_B=1/(float_t)left_census.census_hanming_dist(CensusLeftB.at<double>(i,j),CensusRightB.at<double>(i,current_right));
                    float census_hanming_G=1/(float_t)left_census.census_hanming_dist(CensusLeftG.at<double>(i,j),CensusRightG.at<double>(i,current_right));
                    float census_hanming_R=1/(float_t)left_census.census_hanming_dist(CensusLeftR.at<double>(i,j),CensusRightR.at<double>(i,current_right));
                    float aver_census=(census_hanming_B+census_hanming_G+census_hanming_R)/3;
                    float sum_abs_census=abs(census_hanming_B-aver_census)+abs(census_hanming_G-aver_census)+abs(census_hanming_R-aver_census);
                    if(sum_abs_census!=0){
                        CT_weight[0]=abs(census_hanming_B-aver_census)/sum_abs_census;
                        CT_weight[1]=abs(census_hanming_G-aver_census)/sum_abs_census;
                        CT_weight[2]=abs(census_hanming_R-aver_census)/sum_abs_census;
                    }
                    else{
                        CT_weight[0]=0.3;
                        CT_weight[1]=0.3;
                        CT_weight[2]=0.3;
                    }
                    float diss_corr=CO_weight[0]/diss0+CO_weight[1]/diss1+CO_weight[2]/diss2;
                    float diss_ct=CT_weight[0]/census_hanming_B+CT_weight[1]/census_hanming_G+CT_weight[2]/census_hanming_R;
                    float diss=0.05*diss_corr+0.5*diss_ct;
                    // printf("diss : %f \n",diss);
                    if(isnan(diss))
                        cost_pix.push_back(FLT_MAX/2);
                    else
                        cost_pix.push_back(diss);
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
    // 代价聚合
    cuda_main(cost_ini,cost_agg,rows,cols,D,image);     // cuda keral
    // 视差计算
    ComputeDisparity();
    // 优化视差
    MedianFilter(DispLinerImage,DispLinerImage,left.cols,left.rows,7);   //  the disp result
    // to convert the shape to w*h
//    for(int i=0;i<left.rows;i++){
//        for(int j=0;j<left.cols;j++){
//            disp_map.data[i*left.cols+j]=DispLinerImage[i*left.cols+j];
//        }
//    }
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
            }
            else {
                disp_image.at<uchar>(i,j) = static_cast<uchar>((disp - min_disp) / (max_disp - min_disp) * 255);
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
    //////  end ///////
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
