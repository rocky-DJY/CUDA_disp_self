#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <sys/time.h>
#include <stdio.h>
#include <math.h>
#include <iostream>
using namespace std;
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
__global__ void AggreKeral_left(float ***Coss_disp,float ***res,int cols,int D)
{
    // Coss_dosp<input> the disprity coss of the left and right image
    // cols<input> the number of  each row's whole point
    // D is the nums of every point's disp range
    int tid=threadIdx.x;
    // start Aggregation
    for(int i=0;i<cols;i++)
        for(int j=0;j<D;j++){
            res[tid][i][j]=Coss_disp[tid][i][j];
        }
}

extern "C" int cuda_main()
{
    int rows;
    int cols;
    int D;
    __device__ float ***res;
    ////////////////////  data init //////////////////////////////
    double ***f_3 = (double***)malloc(rows * sizeof(double***));
    double **f_2 = (double**)malloc(rows * cols * sizeof(double**));
    double *f_1 = (double*)malloc(rows*cols * D * sizeof(double*));

    double ***d_3;
    cudaMalloc((void**)(&d_3), rows * sizeof(double***));
    double **d_2;
    cudaMalloc((void**)(&d_2), rows*cols * sizeof(double**));
    double *d_1;
    cudaMalloc((void**)(&d_1), rows*cols * D * sizeof(double));

    for (int i = 0; i < rows*cols * D; i++)// ini the data 3D
    {
        f_1[i] = 0;
    }
    cudaMemcpy(d_1, f_1, rows*cols * D * sizeof(double), cudaMemcpyHostToDevice);
    for (int i = 0; i < rows*cols ; i++)
    {
        f_2[i] = d_1 + i * D;
    }
    cudaMemcpy(d_2, f_2, rows*cols * sizeof(double**), cudaMemcpyHostToDevice);
    for (int i = 0; i < rows; i++)
    {
        f_3[i] = d_2 + cols * i;
    }
    cudaMemcpy(d_3, f_3, rows * sizeof(double***), cudaMemcpyHostToDevice);

    /////////////////////////////////////////////////////
    // malloc the result space on device
    float ***Agg_res;
    cudaMalloc((void**)&res, sizeof(int **) * rows); // int **
    // malloc the host space val
    float ***Coss_Host;  // Host 3 dim vector coss_disp
    Coss_Host=(float***)malloc(sizeof(int*)*rows);
    // def keral   left aggregation;
    dim3 grid(1);
    dim3 block(1024);
    AggreKeral_left<<<grid,block>>>(Coss_Host,res,cols,D);
    return 0;
}