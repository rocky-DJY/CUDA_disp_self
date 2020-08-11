#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
using namespace std;
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
__global__ void AggreKeral_left(float ***Coss_disp,float ***res,int cols,int D)
{
    // Coss_dosp<input> the disprity coss of the left and right image
    // cols<input> the number of  each row's whole point
    // D is the nums of every point's disp range
    // the first point as the L(p,d) init
    // range from 28 to 1051
    int tid=threadIdx.x;
    res[tid+28][0]=Coss_disp[tid+28][0];
    for(int i=1;i<cols;i++) { // point index start 1:cols-1
        // step 1 find the min cost in the L(p-1,i) // i stand the D range
        float min_lpd=res[tid+28][i-1][0];
        for(int m=1;m<D;m++){
            if(res[tid+28][i-1][m]<min_lpd)
                min_lpd=res[tid+28][i-1][m];
        }
        // step 2 compute different disp aggregation
        for (int j = 0; j < D; j++) { // Disp

            res[tid + 28][i][j] = Coss_disp[tid][i][j];
        }
    }
}

extern "C" int cuda_main(vector<vector<vector<float_t> > > cost_disp)
{
    int rows;  // cost_disp  first point
    int cols;  // second
    int D;     // third
    ////////////////////  data init //////////////////////////////
    float ***f_3 = (float***)malloc(rows * sizeof(float***));
    float ***res_3 = (float***)malloc(rows * sizeof(float***));
    float **f_2 = (float**)malloc(rows * cols * sizeof(float**));
    float **res_2 = (float**)malloc(rows * cols * sizeof(float**));
    float *f_1 = (float*)malloc(rows*cols * D * sizeof(float*));
    float *res_1 = (float*)malloc(rows*cols * D * sizeof(float*));

    float ***d_res_3;  // result disp after aggregation
    float ***d_3;      // initional disp
    cudaMalloc((void**)(&d_3), rows * sizeof(float***));
    cudaMalloc((void**)(&d_res_3), rows * sizeof(float***));
    float **d_2;
    float **d_res_2;
    cudaMalloc((void**)(&d_2), rows*cols * sizeof(float**));
    cudaMalloc((void**)(&d_res_2), rows*cols * sizeof(float**));
    float *d_1;
    float *d_res_1;
    cudaMalloc((void**)(&d_1), rows*cols * D * sizeof(float));
    cudaMalloc((void**)(&d_res_1), rows*cols * D * sizeof(float));

//    for (int i = 0; i < rows*cols * D; i++)// ini the data 3D
//    {
//        f_1[i] = 0;
//    }
    int index=0;
    for(int i=0;i<rows;i++){
        for(int j=0;i<cols;j++){
            for(int k=0;k<D;k++){
                f_1[index++]=cost_disp[i][j][k];
            }
        }
    }
    cudaMemcpy(d_1, f_1, rows*cols * D * sizeof(float), cudaMemcpyHostToDevice);
    for (int i = 0; i < rows*cols ; i++)
    {
        f_2[i] = d_1 + i * D;
        res_2[i] = d_res_1 + i * D;
    }
    cudaMemcpy(d_2, f_2, rows*cols * sizeof(float**), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_2, res_2, rows*cols * sizeof(float**), cudaMemcpyHostToDevice);
    for (int i = 0; i < rows; i++)
    {
        f_3[i] = d_2 + cols * i;
        res_3[i] = d_res_2 + cols * i;
    }
    cudaMemcpy(d_3, f_3, rows * sizeof(float***), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_3, res_3, rows * sizeof(float***), cudaMemcpyHostToDevice);
    //////////////////////////// end /////////////////////////

    // def keral   left aggregation;
    dim3 grid(1);
    dim3 block(1024);
    AggreKeral_left<<<grid,block>>>(d_3,d_res_3,cols,D);
    return 0;
}