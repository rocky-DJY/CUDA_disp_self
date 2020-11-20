//
// Created by maxwell on 10/26/20.
//
#include <chrono>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace chrono;
#define image_cols 1920
#define image_rows 1080
// kernel function for census transform
__global__ void census_transform(uint* inputimage,double* outputimage,int windows_sizex,int windows_sizey,float threshold_val){
    int col_index=threadIdx.x + blockDim.x * blockIdx.x;
    int row_index=blockDim.y*blockIdx.y+threadIdx.y;
    int offsetx=(windows_sizex-1)/2;
    int offsety=(windows_sizey-1)/2;
    uint64 census = 0u;    // unsigned long
    double sum_kernal=0;
    //   从顶点出发
    if((col_index<image_cols-windows_sizex)&&(row_index<image_rows-windows_sizey)){
        // printf("col:%d,rows:%d",col_index,row_index);
        // 取中心像素
        uint8_t curr_middle=inputimage[(row_index+offsety)*image_cols+(col_index+offsetx)];
        // 求块内平均值
        for(int i=row_index;i<row_index+windows_sizey;i++){
            for(int j=col_index;j<col_index+windows_sizex;j++){
                sum_kernal+=inputimage[i*image_cols+j];
            }
        }
        double  aver=sum_kernal/(windows_sizex*windows_sizey);
        // 遍历块
        for(int i=row_index;i<row_index+windows_sizey;i++){
            for(int j=col_index;j<col_index+windows_sizex;j++){
                if(!(i==row_index+offsety&j==col_index+offsetx)){
                    uint8_t curr=inputimage[i*image_cols+j];
                    census = census << 1;
                    if(curr<curr_middle){
                        census+=1;
                    }
                    census=census<<1;
                    if(curr-aver<threshold_val){
                        census+=1;
                    }
                }
            }
            outputimage[(row_index+offsety)*image_cols+(col_index+offsetx)]=census;
        }
    }
    else{
        outputimage[row_index*image_cols+col_index]=0;
    }
}
extern  "C" void transform(const cv::Mat input_image, cv::Mat &modified_image, int window_sizex, int window_sizey, float threshold_val) {
    int rows=input_image.size().height;
    int cols=input_image.size().width;
    modified_image = cv::Mat::zeros(rows, cols, CV_64FC1);  // 64   census result
    //  host memory malloc
    uint *image_host = (uint *)malloc(sizeof(uint) * rows * cols);
    double *modified_image_host = (double *)malloc(sizeof(double) * rows * cols);
    // device memory malloc
    uint *image_device;
    double *modified_image_device;
    cudaMalloc((void**)(&image_device), rows*cols * sizeof(uint));
    cudaMalloc((void**)(&modified_image_device), rows*cols * sizeof(double));
    // copy the image src to one dimension pointer
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            image_host[i*cols+j]=(int)input_image.at<u_char>(i,j);
        }
    }
    for(int i=0;i<rows*cols;i++){
        modified_image_host[i]=0;
    }
    cudaMemcpy(image_device,image_host,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    // cout<<"error: "<<error<<endl;
    // 32*32 threads unity is a block
    dim3 block_size(32,32);
    dim3 grid_size( (cols + block_size.x - 1)/ block_size.x, (rows + block_size.y - 1) / block_size.y );
    // cout<<grid_size.x<<" "<<grid_size.y<<endl;
    // cout<<"enter cuda census transform "<<endl;
    census_transform<<<grid_size,block_size>>>(image_device,modified_image_device,window_sizex,window_sizey,threshold_val);
    cudaDeviceSynchronize();
    cudaError_t cudaStatus=cudaGetLastError();
    if(cudaStatus!=cudaSuccess){
        cout<<"cudastatus: "<<cudaStatus<<endl;
        fprintf(stderr,"census_transform failed:%s\n",cudaGetErrorString(cudaStatus));
    }
    // download the data from the device to host
    cudaMemcpy(modified_image_host,modified_image_device,rows*cols*sizeof(double),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            modified_image.at<double>(i,j)=modified_image_host[i*cols+j];
        }
    }
    delete image_host;
    delete modified_image_host;
    cudaFree(image_device);
    cudaFree(modified_image_device);
}
