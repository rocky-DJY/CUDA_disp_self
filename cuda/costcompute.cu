//
// Created by maxwell on 10/28/20.
//
#include <chrono>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#define disp_range 224
#define min_disp 10
#define image_cols 1920
#define image_rows 1080
#define winsize 5
using namespace std;

__global__ void cost_kernal(
        float* left_R_device,float* left_G_device,float* left_B_device,float* right_R_device,float* right_G_device,float* right_B_device,
        double* left_ct_R_device,double* left_ct_G_device,double* left_ct_B_device,double* right_ct_R_device,double* right_ct_G_device,double* right_ct_B_device,
        float* cost_res_device){
    int col_index=blockDim.x*blockIdx.x+threadIdx.x;
    int row_index=blockDim.y*blockIdx.y+threadIdx.y;
    int offset=(winsize-1)/2;
    if((col_index>offset)&&(col_index<image_cols-offset)&&(row_index>offset)&&(row_index<image_rows-offset)){
        ///*** census data ***///
        long long pl[3];
        pl[0]=left_ct_B_device[row_index*image_cols+col_index];
        pl[1]=left_ct_G_device[row_index*image_cols+col_index];
        pl[2]=left_ct_R_device[row_index*image_cols+col_index];
        ///*** RGB data ***///
        for(int i=10;i<disp_range+10;i++){
            int right_col_index=col_index-(i);  // current right col index
            if(right_col_index>=offset){
                ///////////**** census hanming distence compute  ************/////////////
                float census_hanming[3];
                float CT_weight[3];
                census_hanming[0]=0;
                census_hanming[1]=0;
                census_hanming[2]=0;
                long long pr[3];
                long long v[3];
                pr[0]=right_ct_B_device[row_index*image_cols+right_col_index];
                pr[1]=right_ct_G_device[row_index*image_cols+right_col_index];
                pr[2]=right_ct_R_device[row_index*image_cols+right_col_index];
                v[0]=pl[0]^pr[0];
                v[1]=pl[1]^pr[1];
                v[2]=pl[2]^pr[2];
                while(v[0]){
                    v[0]&=(v[0]-1);
                    census_hanming[0]++;
                }
                // __syncthreads();
                while(v[1]){
                    v[1]&=(v[1]-1);
                    census_hanming[1]++;
                }
                while(v[2]){
                    v[2]&=(v[2]-1);
                    census_hanming[2]++;
                }
                float aver_census=(census_hanming[0]+census_hanming[1]+census_hanming[2])/3;
                float BB_diss=exp(-abs(census_hanming[0]-aver_census));
                float GG_diss=exp(-abs(census_hanming[1]-aver_census));
                float RR_diss=exp(-abs(census_hanming[2]-aver_census));
                CT_weight[0]=BB_diss/(BB_diss+GG_diss+RR_diss);
                CT_weight[1]=GG_diss/(BB_diss+GG_diss+RR_diss);
                CT_weight[2]=RR_diss/(BB_diss+GG_diss+RR_diss);
                float diss_ct=CT_weight[0]*census_hanming[0]+CT_weight[1]*census_hanming[1]+CT_weight[2]*census_hanming[2];
                //////////****end****//////////
                float diss=diss_ct;
                if(isnan(diss)){
                    cost_res_device[row_index*image_cols*disp_range+col_index*disp_range+i-10]=FLT_MAX/2;
                }
                else{
                    cost_res_device[row_index*image_cols*disp_range+col_index*disp_range+i-10]=diss;
                }
            }
            else{
                cost_res_device[row_index*image_cols*disp_range+col_index*disp_range+i-10]=FLT_MAX/2;
            }
        }
    }
}

extern  "C" void cost_cuda_main(vector<vector<vector<float>>> &cost_nums,
        double* L_census_R,double* L_census_G,double* L_census_B,
        double* R_census_R,double* R_census_G,double* R_census_B,
        cv::Mat left,cv::Mat right){
// left and right is the RGB image matrix ,which should to split into 3
// census_r,census_g and census_b are 1 channel census transform result matrix;
// first to transform the matrix to 1 dim array;
    int cols=left.cols;
    int rows=left.rows;
    cv::Mat left_RGB[3],right_RGB[3];
    cv::split(left ,left_RGB );
    cv::split(right,right_RGB);
    ////host memory molloc/////////
    float* left_R_host= (float *)malloc(sizeof(float) * rows * cols);
    float* left_G_host= (float *)malloc(sizeof(float) * rows * cols);
    float* left_B_host= (float *)malloc(sizeof(float) * rows * cols);
    float* right_R_host= (float *)malloc(sizeof(float) * rows * cols);
    float* right_G_host= (float *)malloc(sizeof(float) * rows * cols);
    float* right_B_host= (float *)malloc(sizeof(float) * rows * cols);
    float* cost_res_host=(float *)malloc(sizeof(float)*rows*cols*disp_range);
    ////**end**/////
    // device memory molloc
    float* left_R_device;
    cudaMalloc((void**)(&left_R_device), rows*cols * sizeof(float));
    float* left_G_device;
    cudaMalloc((void**)(&left_G_device), rows*cols * sizeof(float));
    float* left_B_device;
    cudaMalloc((void**)(&left_B_device), rows*cols * sizeof(float));
    float* right_R_device;
    cudaMalloc((void**)(&right_R_device), rows*cols * sizeof(float));
    float* right_G_device;
    cudaMalloc((void**)(&right_G_device), rows*cols * sizeof(float));
    float* right_B_device;
    cudaMalloc((void**)(&right_B_device), rows*cols * sizeof(float));

    double* left_ct_R_divice;
    cudaMalloc((void**)(&left_ct_R_divice), rows*cols * sizeof(double));

    double* left_ct_G_divice;
    cudaMalloc((void**)(&left_ct_G_divice), rows*cols * sizeof(double));

    double* left_ct_B_divice;
    cudaMalloc((void**)(&left_ct_B_divice), rows*cols * sizeof(double));

    double* right_ct_R_divice;
    cudaMalloc((void**)(&right_ct_R_divice), rows*cols * sizeof(double));

    double* right_ct_G_divice;
    cudaMalloc((void**)(&right_ct_G_divice), rows*cols * sizeof(double));

    double* right_ct_B_divice;
    cudaMalloc((void**)(&right_ct_B_divice), rows*cols * sizeof(double));

    float* cost_res_device;
    cudaMalloc((void**)(&cost_res_device), disp_range*rows*cols * sizeof(float));
    //*end*//
    ///**host data initial**///
    int index=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            left_B_host[index]=(float)left_RGB[0].at<uchar>(i,j);
            left_G_host[index]=(float)left_RGB[1].at<uchar>(i,j);
            left_R_host[index]=(float)left_RGB[2].at<uchar>(i,j);
            right_B_host[index]=(float)right_RGB[0].at<uchar>(i,j);
            right_G_host[index]=(float)right_RGB[1].at<uchar>(i,j);
            right_R_host[index]=(float)right_RGB[2].at<uchar>(i,j);
            // printf("%lf\n",L_census_R[index]);
            index++;
        }
    }
    ///**end**///
    // memory copy //
    cudaMemcpy(left_ct_R_divice,L_census_R,rows*cols*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(left_ct_G_divice,L_census_G,rows*cols*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(left_ct_B_divice,L_census_B,rows*cols*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(right_ct_B_divice,R_census_B,rows*cols*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(right_ct_G_divice,R_census_G,rows*cols*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(right_ct_R_divice,R_census_R,rows*cols*sizeof(double),cudaMemcpyHostToDevice);

    cudaMemcpy(left_R_device,left_R_host,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(left_G_device,left_G_host,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(left_B_device,left_B_host,rows*cols*sizeof(float),cudaMemcpyHostToDevice);

    cudaMemcpy(right_B_device,right_B_host,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(right_G_device,right_G_host,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(right_R_device,right_R_host,rows*cols*sizeof(float),cudaMemcpyHostToDevice);
    //***end***//
    cudaDeviceSynchronize();  //
    dim3 block_size(32,32);
    dim3 grid_size((cols + block_size.x - 1)/ block_size.x,(rows + block_size.y - 1)/ block_size.y);
    cout<<"cost keral size: "<<grid_size.x<<" "<<grid_size.y<<endl;
    cost_kernal<<<grid_size,block_size>>>(
            left_R_device,left_G_device,left_B_device,right_R_device,right_G_device,right_B_device,
            left_ct_R_divice,left_ct_G_divice,left_ct_B_divice,right_ct_R_divice,right_ct_G_divice,right_ct_B_divice,
            cost_res_device);
    cudaDeviceSynchronize();
    cudaMemcpy(cost_res_host,cost_res_device,disp_range*rows*cols *sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();  //
    cudaError_t cudaStatus=cudaGetLastError();
    if(cudaStatus!=cudaSuccess){
        cout<<"cudastatus: "<<cudaStatus<<endl;
        fprintf(stderr,"cost_computer failed:%s\n",cudaGetErrorString(cudaStatus));
    }
    int offset_x=(winsize-1)/2;
    int offset_y=(winsize-1)/2;
    for(int i=offset_y;i<rows-offset_y;i++){
        for(int j=offset_x;j<cols-offset_x;j++){
            for(int k=0;k<disp_range;k++){
                cost_nums[i-offset_y][j-offset_x][k]=cost_res_host[i*cols*disp_range+j*disp_range+k];
            }
        }
    }
    // release  the memory
    free(left_B_host);
    free(left_G_host);
    free(left_R_host);
    free(right_B_host);
    free(right_G_host);
    free(right_R_host);
    free(cost_res_host);
    cudaFree(left_B_device);
    cudaFree(left_G_device);
    cudaFree(left_R_device);
    cudaFree(right_B_device);
    cudaFree(right_G_device);
    cudaFree(right_R_device);
    cudaFree(left_ct_B_divice);
    cudaFree(left_ct_G_divice);
    cudaFree(left_ct_R_divice);
    cudaFree(right_ct_B_divice);
    cudaFree(right_ct_G_divice);
    cudaFree(right_ct_R_divice);
    cudaFree(cost_res_device);
    ///***** end ******///
}
