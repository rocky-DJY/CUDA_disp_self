#include <chrono>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace chrono;
#define P1_ini 10       // 10
#define P2_ini 125     // 125
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
__global__ void AggreKeral_left(float *Coss_disp,float *res,uint *image,int rows,int cols,int D)
{
    // Coss_dosp<input> the disprity coss of the left and right image
    // cols<input> the number of  each row's whole point
    // D is the nums of every point's disp range
    // the first point as the L(p,d) init
    int id_rows=blockIdx.x;
    int id_d=threadIdx.x;
    float min_last_path;
    float p1=P1_ini;
    float p2=P2_ini;
    float min_agg;
    float l_p0=0;   // stand l(p-r,d)
    float l_p1=0;   // stand l(p-r,d-1)+p1
    float l_p2=0;   // stand l(p-r,d+1)+p1
    float l_p3=0;   // stand min_last_path+p2
    //res[id_rows][0][id_d]=Coss_disp[id_rows][0][id_d];      // init the l(p,r)
    res[id_rows*cols*D+id_d]=Coss_disp[id_rows*cols*D+id_d];  // init the l(p,r)
    __syncthreads();
    float last_Lp[3];
    if(id_d==0){
        last_Lp[0]=res[id_rows*cols*D+id_d+1];
        last_Lp[1]=res[id_rows*cols*D+id_d];     //res[id_rows][0][id_d];
        last_Lp[2]=res[id_rows*cols*D+id_d+2];
    }
    if(id_d==D-1){
        last_Lp[0]=res[id_rows*cols*D+id_d-1];
        last_Lp[1]=res[id_rows*cols*D+id_d];
        last_Lp[2]=res[id_rows*cols*D+id_d-2];
    }
    if(id_d!=0&id_d!=D-1){
        last_Lp[0]=res[id_rows*cols*D+id_d-1];
        last_Lp[1]=res[id_rows*cols*D+id_d];
        last_Lp[2]=res[id_rows*cols*D+id_d+1];
    }
    __syncthreads();
    for(int i=1;i<cols;i++){
        // step 1 find min cost in last path
        min_last_path=res[id_rows*cols*D+(i-1)*D];      // res[id_rows][i-1][0];
        for(int j=1;j<D;j++){
            if(res[id_rows*cols*D+(i-1)*D+j]<min_last_path){
                min_last_path=res[id_rows*cols*D+(i-1)*D+j];
            }
        }
        l_p0=last_Lp[1];
        l_p1=last_Lp[0]+p1;
        l_p2=last_Lp[2]+p1;
        l_p3=min_last_path+p2/(image[id_rows*cols+i]-image[id_rows*cols+i-1]);
        //step 2
        min_agg=l_p0;
        if(l_p1<min_agg)
            min_agg=l_p1;
        if(l_p2<min_agg)
            min_agg=l_p2;
        if(l_p3<min_agg)
            min_agg=l_p3;
        res[id_rows*cols*D+i*D+id_d]=Coss_disp[id_rows*cols*D+i*D+id_d]+min_agg-min_last_path;
        __syncthreads();
        if(id_d==0){
            last_Lp[0]=res[id_rows*cols*D+i*D+id_d+1];
            last_Lp[1]=res[id_rows*cols*D+i*D+id_d];    // [id_rows][i][id_d];
            last_Lp[2]=res[id_rows*cols*D+i*D+id_d+2];
        }
        if(id_d==D-1){
            last_Lp[0]=res[id_rows*cols*D+i*D+id_d];
            last_Lp[1]=res[id_rows*cols*D+i*D+id_d-1];
            last_Lp[2]=res[id_rows*cols*D+i*D+id_d-2];
        }
        if(id_d!=0&id_d!=D-1){
            last_Lp[0]=res[id_rows*cols*D+i*D+id_d-1];
            last_Lp[1]=res[id_rows*cols*D+i*D+id_d];
            last_Lp[2]=res[id_rows*cols*D+i*D+id_d+1];
        }
        __syncthreads();
    }
}
__global__ void AggreKeral_right(float *Coss_disp,float *res,uint *image,int rows,int cols,int D){
    // Coss_dosp<input> the disprity coss of the left and right image
    // cols<input> the number of  each row's whole point
    // D is the nums of every point's disp range
    // the first point as the L(p,d) init
    int id_rows=blockIdx.x;
    int id_d=threadIdx.x;
    float p1=P1_ini;
    float p2=P2_ini;
    float min_last_path;
    float last_LP[3];
    float min_agg;
    float l_p0=0;   // stand l(p-r,d)
    float l_p1=0;   // stand l(p-r,d-1)+p1
    float l_p2=0;   // stand l(p-r,d+1)+p1
    float l_p3=0;   // stand min_last_path+p2
    res[id_rows*cols*D+(cols-1)*D+id_d]=Coss_disp[id_rows*cols*D+(cols-1)*D+id_d];  // init the l(p,r)
    __syncthreads();
    if(id_d==0){
        last_LP[0]=res[id_rows*cols*D+(cols-1)*D+ id_d];   //res[id_rows][cols-1][id_d];
        last_LP[1]=res[id_rows*cols*D+(cols-1)*D+ id_d+1]; //res[id_rows][cols-1][id_d+1];
        last_LP[2]=res[id_rows*cols*D+(cols-1)*D+ id_d+2]; //res[id_rows][cols-1][id_d+2];
    }
    if(id_d==D-1){
        last_LP[0]=res[id_rows*cols*D+(cols-1)*D+ id_d];
        last_LP[1]=res[id_rows*cols*D+(cols-1)*D+ id_d-1];
        last_LP[2]=res[id_rows*cols*D+(cols-1)*D+ id_d-2];
    }
    if(id_d!=0&id_d!=D-1){
        last_LP[0]=res[id_rows*cols*D+(cols-1)*D+ id_d-1];
        last_LP[1]=res[id_rows*cols*D+(cols-1)*D+ id_d];
        last_LP[2]=res[id_rows*cols*D+(cols-1)*D+ id_d+1];
    }
    for(int i=cols-2;i>-1;i--){
        // step 1 find min cost in last path
        // min_last_path=res[id_rows][i+1][0];
        min_last_path=res[id_rows*cols*D+(i+1)*D];
        for(int j=1;j<D;j++){
            if(res[id_rows*cols*D+(i+1)*D+j]<min_last_path){
                //min_last_path=res[id_rows][i-1][j];
                min_last_path=res[id_rows*cols*D+(i+1)*D+j];
            }
        }
        l_p0=last_LP[1];
        l_p1=last_LP[0]+p1;
        l_p2=last_LP[2]+p1;
        l_p3=min_last_path+p2/(image[id_rows*cols+i]-image[id_rows*cols+i+1]);
        // step 2
        min_agg=l_p0;
        if(l_p1<min_agg)
            min_agg=l_p1;
        if(l_p2<min_agg)
            min_agg=l_p2;
        if(l_p3<min_agg)
            min_agg=l_p3;
        //res[id_rows][i][id_d]=Coss_disp[id_rows][i][id_d]+min_agg-min_last_path;
        res[id_rows*cols*D+i*D+id_d]=Coss_disp[id_rows*cols*D+i*D+id_d]+min_agg-min_last_path;
        __syncthreads();
        if(id_d==0){
            last_LP[0]=res[id_rows*cols*D+i*D+ id_d];   //res[id_rows][0][id_d];
            last_LP[1]=res[id_rows*cols*D+i*D+ id_d+1]; //res[id_rows][0][id_d+1];
            last_LP[2]=res[id_rows*cols*D+i*D+ id_d+2]; //res[id_rows][0][id_d+2];
        }
        if(id_d==D-1){
            last_LP[0]=res[id_rows*cols*D+i*D+ id_d];
            last_LP[1]=res[id_rows*cols*D+i*D+ id_d-1];
            last_LP[2]=res[id_rows*cols*D+i*D+ id_d-2];
        }
        if(id_d!=0&id_d!=D-1){
            last_LP[0]=res[id_rows*cols*D+i*D+ id_d-1];
            last_LP[1]=res[id_rows*cols*D+i*D+ id_d];
            last_LP[2]=res[id_rows*cols*D+i*D+ id_d+1];
        }
        __syncthreads();
    }
}
__global__ void Aggrekeral_top(float *Coss_disp,float *res,uint *image,int rows,int cols,int D){  // top to down
    int id_cols=blockIdx.x;
    int id_d=threadIdx.x;
    float p1=P1_ini;
    float p2=P2_ini;
    float min_last_path;
    float last_LP[3];
    float min_agg;
    float l_p0=0;   // stand l(p-r,d)
    float l_p1=0;   // stand l(p-r,d-1)+p1
    float l_p2=0;   // stand l(p-r,d+1)+p1
    float l_p3=0;   // stand min_last_path+p2
    res[id_cols*D+id_d]=Coss_disp[id_cols*D+id_d];  // init the l(p,r)
    __syncthreads();
    if(id_d==0){
        last_LP[0]=res[id_cols*D+ id_d];
        last_LP[1]=res[id_cols*D+ id_d+1];
        last_LP[2]=res[id_cols*D+ id_d+2];
    }
    if(id_d==D-1){
        last_LP[0]=res[id_cols*D+ id_d];
        last_LP[1]=res[id_cols*D+ id_d-1];
        last_LP[2]=res[id_cols*D+ id_d-2];
    }
    if(id_d!=0&id_d!=D-1){
        last_LP[0]=res[id_cols*D+ id_d-1];
        last_LP[1]=res[id_cols*D+ id_d];
        last_LP[2]=res[id_cols*D+ id_d+1];
    }
    __syncthreads();
    for(int i=1;i<rows;i++){
        // step 1 find min cost in last path
        min_last_path=res[(i-1)*cols*D+id_cols*D];
        for(int j=1;j<D;j++){
            if(res[(i-1)*cols*D+id_cols*D+j]<min_last_path){
                //min_last_path=res[i-1][j][k];
                min_last_path=res[(i-1)*cols*D+id_cols*D+j];
            }
        }
        l_p0=last_LP[1];
        l_p1=last_LP[0]+p1;
        l_p2=last_LP[2]+p1;
        l_p3=min_last_path+p2/(image[i*cols+id_cols]-image[(i-1)*cols+id_cols]);
        // step 2
        min_agg=l_p0;
        if(l_p1<min_agg)
            min_agg=l_p1;
        if(l_p2<min_agg)
            min_agg=l_p2;
        if(l_p3<min_agg)
            min_agg=l_p3;
        res[i*cols*D+id_cols*D+id_d]=Coss_disp[i*cols*D+id_cols*D+id_d]+min_agg-min_last_path;
        __syncthreads();
        if(id_d==0){
            last_LP[0]=res[i*cols*D+id_cols*D+ id_d];   //res[i][id_cols][id_d];
            last_LP[1]=res[i*cols*D+id_cols*D+ id_d+1]; //res[i][id_cols][id_d+1];
            last_LP[2]=res[i*cols*D+id_cols*D+ id_d+2]; //res[i][id_cols][id_d+2];
        }
        if(id_d==D-1){
            last_LP[0]=res[i*cols*D+id_cols*D+ id_d];
            last_LP[1]=res[i*cols*D+id_cols*D+ id_d-1];
            last_LP[2]=res[i*cols*D+id_cols*D+ id_d-2];
        }
        if(id_d!=0&id_d!=D-1){
            last_LP[0]=res[i*cols*D+id_cols*D+ id_d-1];
            last_LP[1]=res[i*cols*D+id_cols*D+ id_d];
            last_LP[2]=res[i*cols*D+id_cols*D+ id_d+1];
        }
        __syncthreads();
    }
}
__global__ void Aggrekeral_down(float *Coss_disp,float *res,uint *image,int rows,int cols,int D){  // down to top
    int id_cols=blockIdx.x;
    int id_d=threadIdx.x;
    float p1=P1_ini;
    float p2=P2_ini;
    float min_last_path;
    float last_LP[3];
    float min_agg;
    float l_p0=0;   // stand l(p-r,d)
    float l_p1=0;   // stand l(p-r,d-1)+p1
    float l_p2=0;   // stand l(p-r,d+1)+p1
    float l_p3=0;   // stand min_last_path+p2
    res[(rows-1)*cols*D+id_cols*D+id_d]=Coss_disp[(rows-1)*cols*D+id_cols*D+id_d]; // init the l(p,r)
    __syncthreads();
    if(id_d==0){
        last_LP[0]=res[(rows-1)*cols*D+id_cols*D+ id_d];
        last_LP[1]=res[(rows-1)*cols*D+id_cols*D+ id_d+1];
        last_LP[2]=res[(rows-1)*cols*D+id_cols*D+ id_d+2];
    }
    if(id_d==D-1){
        last_LP[0]=res[(rows-1)*cols*D+id_cols*D+ id_d];
        last_LP[1]=res[(rows-1)*cols*D+id_cols*D+ id_d-1];
        last_LP[2]=res[(rows-1)*cols*D+id_cols*D+ id_d-2];
    }
    if(id_d!=0&id_d!=D-1){
        last_LP[0]=res[(rows-1)*cols*D+id_cols*D+ id_d-1];
        last_LP[1]=res[(rows-1)*cols*D+id_cols*D+ id_d];
        last_LP[2]=res[(rows-1)*cols*D+id_cols*D+ id_d+1];
    }
    for(int i=rows-2;i>-1;i--){
        // step 1 find min cost in last path
        min_last_path=res[(i+1)*cols*D+id_cols*D];
        for(int j=1;j<D;j++){
            if(res[(i+1)*cols*D+id_cols*D+j]<min_last_path){
                //min_last_path=res[i-1][j][k];
                min_last_path=res[(i+1)*cols*D+id_cols*D+j];
            }
        }
        l_p0=last_LP[1];
        l_p1=last_LP[0]+p1;
        l_p2=last_LP[2]+p1;
        l_p3=min_last_path+p2/(image[i*cols+id_cols]-image[(i+1)*cols+id_cols]);
        // step 2
        min_agg=l_p0;
        if(l_p1<min_agg)
            min_agg=l_p1;
        if(l_p2<min_agg)
            min_agg=l_p2;
        if(l_p3<min_agg)
            min_agg=l_p3;
        res[i*cols*D+id_cols*D+id_d]=Coss_disp[i*cols*D+id_cols*D+id_d]+min_agg-min_last_path;
        __syncthreads();
        if(id_d==0){
            last_LP[0]=res[i*cols*D+id_cols*D+ id_d];
            last_LP[1]=res[i*cols*D+id_cols*D+ id_d+1];
            last_LP[2]=res[i*cols*D+id_cols*D+ id_d+2];
        }
        if(id_d==D-1){
            last_LP[0]=res[i*cols*D+id_cols*D+ id_d];
            last_LP[1]=res[i*cols*D+id_cols*D+ id_d-1];
            last_LP[2]=res[i*cols*D+id_cols*D+ id_d-2];
        }
        if(id_d!=0&id_d!=D-1){
            last_LP[0]=res[i*cols*D+id_cols*D+ id_d-1];
            last_LP[1]=res[i*cols*D+id_cols*D+ id_d];
            last_LP[2]=res[i*cols*D+id_cols*D+ id_d+1];
        }
        __syncthreads();
    }
}
extern "C" int cuda_main( float *cost_disp,float *cost_agg,const int rows,const int cols, const int D,
        const uint *left_image){ // gray image left and right
    cout<<"enter cuda keral..."<<endl;
    //cout<<"size: "<<rows<<","<<cols<<","<<D<<endl;
    ////////////////////  data init //////////////////////////////
    float *f_1      = (float*)malloc(rows*cols * D * sizeof(float*));
    uint  *image    = (uint*)malloc(rows*cols * sizeof(uint*));
    float *d_1;
    float *d_res_1;
    uint  *device_image;
    cudaMalloc((void**)(&d_1), rows*cols * D * sizeof(float));
    cudaMalloc((void**)(&d_res_1), rows*cols * D * sizeof(float));
    cudaMalloc((void**)(&device_image), rows*cols * sizeof(uint));
    memcpy(image,left_image,rows*cols*sizeof(uint));
    // memcpy(f_1,cost_disp,rows*cols*D*sizeof(float));     // src cost copy to f_1;
    cudaMemcpy(device_image,image,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
    cudaMemcpy(d_1, cost_disp, rows*cols * D * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_res_1, res_1, rows*cols * D * sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    ////////////////////////////       end            ///////////////////////
    /////////////////////////// four path aggregation ///////////////////////
    dim3 grid_td(cols);
    dim3 block_td(D);
    // aggregation from top to down
    Aggrekeral_top<<<grid_td,block_td>>> (d_1,d_res_1,device_image,rows,cols,D);
    cudaDeviceSynchronize();
    // aggregation down to top
    Aggrekeral_down<<<grid_td,block_td>>>(d_res_1,d_1,device_image,rows,cols,D);
    cudaDeviceSynchronize();
    dim3 grid(rows);
    dim3 block(D);
    // aggregation from right to left
    AggreKeral_right<<<grid,block>>>     (d_1,d_res_1,device_image,rows,cols,D);
    cudaDeviceSynchronize();
    // aggregation from left to right
    auto start=system_clock::now();
    AggreKeral_left<<<grid,block>>>      (d_res_1,d_1,device_image,rows,cols,D);
    cudaDeviceSynchronize();
    cudaError_t cudaStatus=cudaGetLastError();
    if(cudaStatus!=cudaSuccess){
        fprintf(stderr,"aggregation failed:&s\n",cudaGetErrorString(cudaStatus));
    }
    ////////////////////////// end aggregation //////////////
    /////////// download the data from device ///////////////
    cudaMemcpy(cost_agg,d_1,rows*cols * D * sizeof(float),cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    auto last=system_clock::now();
    auto duration=duration_cast<microseconds>(last-start);
    cout<<"aggregation cost total time:  "
        <<double(duration.count())*microseconds::period::num/microseconds::period::den<<endl;
    /////////// end  copy ////////////
    /////////// free  ///////////
    free(f_1);
    free(image);
    cudaFree(d_1);
    cudaFree(d_res_1);
    cudaFree(device_image);
    /////////// end free /////////////
    cout<<"cuda aggregation done..."<<endl;

    ////////////////////
    return 0;
}