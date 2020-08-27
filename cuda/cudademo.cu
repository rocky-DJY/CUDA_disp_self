#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <sys/time.h>
//#include <device_functions.h>
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
    res[tid+28][0]=Coss_disp[tid+28][0]; // initialize the first points's l(p,d)
    __syncthreads();
    for(int i=1;i<cols;i++) { //Aggregation  point index start 1:cols-1
        // step 1 find the min cost in the L(p-1,i) # i stand the D range
        float min_lpd=res[tid+28][i-1][0];
        for(int m=1;m<D;m++){
            if(res[tid+28][i-1][m]<min_lpd)
                min_lpd=res[tid+28][i-1][m];
        }
        // step 2 compute different disp aggregation cost
        for (int j = 1; j < D; j++) { // Disp j stand the current disp D(j)
            // step 3 find the minimum among ( l(p-1,d), l(p-1,d-1)+p1, l(p-1.d+1)+p1, min_l(p-1,i)+p2 )
            float l1=res[tid+28][i-1][j];      // l(p-1,d)
            float l2=res[tid+28][i-1][j-1];    // l(p-1,d-1)
            float l3=res[tid+28][i-1][j+1];    // l(p-1,d+1)
            float min_poly_L=l1;
            if(l2<min_poly_L)
                min_poly_L=l2;
            if(l3<min_poly_L)
                min_poly_L=l3;
            if(min_lpd<min_poly_L)
                min_poly_L=min_lpd;
            res[tid+28][i][j]=Coss_disp[tid+28][i][j]+min_poly_L-min_lpd;
        }
    }
    __syncthreads();
}
__global__ void AggreKeral_right(float ***Coss_disp,float ***res,int cols,int D){
    // Coss_dosp<input> the disprity coss of the left and right image
    // cols<input> the number of  each row's whole point
    // D is the nums of every point's disp range
    // the first point as the L(p,d) init
    // range from 28 to 1051
    int tid=threadIdx.x;
    res[tid+28][cols-1]=Coss_disp[tid+28][cols-1]; // initialize the first points's l(p,d) right boundaray
    __syncthreads();
    for(int i=cols-2;i>-1;i--) { // point index start cols-2  to 0
        // step 1 find the min cost in the L(p-1,i) # i stand the D range
        float min_lpd=res[tid+28][i+1][0];
        for(int m=1;m<D;m++){
            if(res[tid+28][i+1][m]<min_lpd)
                min_lpd=res[tid+28][i+1][m];
        }
        // step 2 compute different disp aggregation cost
        for (int j = 1; j < D; j++) { // Disp j stand the current disp D(j)
            // step 3 find the minimum among ( l(p-1,d), l(p-1,d-1)+p1, l(p-1.d+1)+p1, min_l(p-1,i)+p2 )
            float l1=res[tid+28][i+1][j];      // l(p-1,d)
            float l2=res[tid+28][i+1][j-1];    // l(p-1,d-1)
            float l3=res[tid+28][i+1][j+1];    // l(p-1,d+1)
            float min_poly_L=l1;
            if(l2<min_poly_L)
                min_poly_L=l2;
            if(l3<min_poly_L)
                min_poly_L=l3;
            if(min_lpd<min_poly_L)
                min_poly_L=min_lpd;
            res[tid+28][i][j]=Coss_disp[tid+28][i][j]+min_poly_L-min_lpd;
        }
    }
    __syncthreads();
}
__global__ void Aggrekeral_top(float ***Coss_disp,float ***res,int rows,int D){  // top to down
     int tid_x=blockIdx.x*blockDim.x+threadIdx.x;    // blockidx point the block coordinate
     res[0][tid_x]=Coss_disp[0][tid_x];              // initinal
     __syncthreads();
     for(int i=1;i<rows;i++) { // point index start  1 to rows-1
        // step 1 find the min cost in the L(p-1,i) # i from the D range
        float min_lpd=res[i-1][tid_x][0];
        for(int m=1;m<D;m++){
            if(res[i-1][tid_x][m]<min_lpd)
                min_lpd=res[i-1][tid_x][m];
        }
        // step 2 compute different disp aggregation cost
        for (int j = 1; j < D; j++) { // Disp j stand the current disp D(j)
            // step 3 find the minimum among ( l(p-1,d), l(p-1,d-1)+p1, l(p-1.d+1)+p1, min_l(p-1,i)+p2 )
            float l1=res[i-1][tid_x][j];      // l(p-1,d)
            float l2=res[i-1][tid_x][j-1];    // l(p-1,d-1)
            float l3=res[i-1][tid_x][j+1];    // l(p-1,d+1)
            float min_poly_L=l1;
            if(l2<min_poly_L)
                min_poly_L=l2;
            if(l3<min_poly_L)
                min_poly_L=l3;
            if(min_lpd<min_poly_L)
                min_poly_L=min_lpd;
            res[i][tid_x][j]=Coss_disp[i][tid_x][j]+min_poly_L-min_lpd;
        }
     }
    __syncthreads();
}
__global__ void Aggrekeral_down(float ***Coss_disp,float ***res,int rows,int D){  // top to down
    int tid_x=blockIdx.x*blockDim.x+threadIdx.x;    // blockidx point the block coordinate
    res[rows-1][tid_x]=Coss_disp[rows-1][tid_x];              // initinal
    __syncthreads();
    for(int i=rows-2;i>-1;i--) { // point index start  1 to rows-1
        // step 1 find the min cost in the L(p-1,i) # i from the D range
        float min_lpd=res[i-1][tid_x][0];
        for(int m=1;m<D;m++){
            if(res[i+1][tid_x][m]<min_lpd)
                min_lpd=res[i+1][tid_x][m];
        }
        // step 2 compute different disp aggregation cost
        for (int j = 1; j < D; j++) { // Disp j stand the current disp D(j)
            // step 3 find the minimum among ( l(p-1,d), l(p-1,d-1)+p1, l(p-1.d+1)+p1, min_l(p-1,i)+p2 )
            float l1=res[i+1][tid_x][j];      // l(p-1,d)
            float l2=res[i+1][tid_x][j-1];    // l(p-1,d-1)
            float l3=res[i+1][tid_x][j+1];    // l(p-1,d+1)
            float min_poly_L=l1;
            if(l2<min_poly_L)
                min_poly_L=l2;
            if(l3<min_poly_L)
                min_poly_L=l3;
            if(min_lpd<min_poly_L)
                min_poly_L=min_lpd;
            res[i][tid_x][j]=Coss_disp[i][tid_x][j]+min_poly_L-min_lpd;
        }
    }
    __syncthreads();
}
extern "C" int cuda_main( vector<vector<vector<float> > > &cost_disp)
{
    cout<<"enter cuda keral..."<<endl;
    int rows=cost_disp.size();  // cost_disp  first index
    int cols=cost_disp[0].size();  // second
    int D=cost_disp[0][0].size();     // third
    cout<<"size: "<<rows<<","<<cols<<D<<endl;

    ////////////////////  data init //////////////////////////////
    float ***f_3   = (float***)malloc(rows * sizeof(float***));
    float ***res_3 = (float***)malloc(rows * sizeof(float***));
    float **f_2    = (float**)malloc(rows * cols * sizeof(float**));
    float **res_2  = (float**)malloc(rows * cols * sizeof(float**));
    float *f_1     = (float*)malloc(rows*cols * D * sizeof(float*));
    float *res_1   = (float*)malloc(rows*cols * D * sizeof(float*));

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
                //cout<<f_1[index++]<<",";
            }
            //cout<<endl;
        }
    }
    cudaMemcpy(d_1, f_1, rows*cols * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_res_1, res_1, rows*cols * D * sizeof(float), cudaMemcpyHostToDevice);
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
    cudaDeviceSynchronize();

    //////////////////////////// end /////////////////////////
    /////////////////////////// four path aggregation ////////di
    dim3 grid(1);
    dim3 block(1024);
    AggreKeral_left<<<grid,block>>>(d_3,d_res_3,cols,D);     // aggregation from left to right
    cudaDeviceSynchronize();                                 // synchronization
    AggreKeral_right<<<grid,block>>>(d_res_3,d_3,cols,D);    // aggregation from right to left
    cudaDeviceSynchronize();
    dim3 top_grid(1);
    int top_block_numOfthreads=cols/2;
    dim3 top_block(top_block_numOfthreads,2);
    Aggrekeral_top<<<top_grid,top_block>>>(d_3,d_res_3,rows,D);  // aggregation from top to down
    cudaDeviceSynchronize();                                     // aggregation from down to top
    Aggrekeral_down<<<top_grid,top_block>>>(d_res_3,d_3,rows,D);
    cudaDeviceSynchronize();
    ////////////////////////// end //////////////////////////
    /////////// download the data from device ///////////////
    // copy the
    cudaMemcpy(d_1,f_1,rows*cols * D * sizeof(float),cudaMemcpyDeviceToHost);
    index=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            for(int k=0;k<D;k++){
                cost_disp[i][j][k]=f_1[index++];
            }
        }
    }
    /////////// end  ////////////
    /////////// free  ///////////
    free(f_3);
    free(f_2);
    free(f_1);
    free(d_3);
    free(d_2);
    free(d_1);
    free(res_3);
    free(res_2);
    free(res_1);
    free(d_res_3);
    free(d_res_2);
    free(d_res_1);
    ////////////////////
    return 0;
}