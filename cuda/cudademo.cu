#include <chrono>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;
using namespace chrono;
#define P1_ini 10;
#define P2_ini 150;
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
__global__ void AggreKeral_left(float ***Coss_disp,float ***res,uint *image,int cols,int D)
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
    res[id_rows][0][id_d]=Coss_disp[id_rows][0][id_d];  // init the l(p,r)
    __syncthreads();
    float last_Lp[3];
    if(id_d==0){
        last_Lp[0]=res[id_rows][0][id_d];
        last_Lp[1]=res[id_rows][0][id_d+1];
        last_Lp[2]=res[id_rows][0][id_d+2];
    }
    if(id_d==D-1){
        last_Lp[0]=res[id_rows][0][id_d];
        last_Lp[1]=res[id_rows][0][id_d-1];
        last_Lp[2]=res[id_rows][0][id_d-2];
    }
    if(id_d!=0&id_d!=D-1){
        last_Lp[0]=res[id_rows][0][id_d-1];
        last_Lp[1]=res[id_rows][0][id_d];
        last_Lp[2]=res[id_rows][0][id_d+1];
    }
    __syncthreads();
    for(int i=1;i<cols;i++){
        // step 1 find min cost in last path
        min_last_path=res[id_rows][i-1][0];
        for(int j=1;j<D;j++){
            if(res[id_rows][i-1][j]<min_last_path){
                min_last_path=res[id_rows][i-1][j];
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
        res[id_rows][i][id_d]=Coss_disp[id_rows][i][id_d]+min_agg-min_last_path;
        if(id_d==0){
            last_Lp[0]=res[id_rows][i][id_d];
            last_Lp[1]=res[id_rows][i][id_d+1];
            last_Lp[2]=res[id_rows][i][id_d+2];
        }
        if(id_d==D-1){
            last_Lp[0]=res[id_rows][i][id_d];
            last_Lp[1]=res[id_rows][i][id_d-1];
            last_Lp[2]=res[id_rows][i][id_d-2];
        }
        if(id_d!=0&id_d!=D-1){
            last_Lp[0]=res[id_rows][i][id_d-1];
            last_Lp[1]=res[id_rows][i][id_d];
            last_Lp[2]=res[id_rows][i][id_d+1];
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
    res[id_rows*cols*D+cols*D+id_d]=Coss_disp[id_rows*cols*D+(cols-1)*D+id_d];  // init the l(p,r)
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
        // min_last_path=res[id_rows][i-1][0];
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
extern "C" int cuda_main( vector<vector<vector<float> > > &cost_disp,
        const cv::Mat left_image){ // gray image left and right
    cout<<"enter cuda keral..."<<endl;
    int rows=cost_disp.size();     // cost_disp  first index
    int cols=cost_disp[0].size();  // second
    int D=cost_disp[0][0].size();  // third
    //cout<<"size: "<<rows<<","<<cols<<","<<D<<endl;
    ////////////////////  data init //////////////////////////////
    float ***f_3   = (float***)malloc(rows * sizeof(float***));
    float ***res_3 = (float***)malloc(rows * sizeof(float***));
    float **f_2    = (float**)malloc(rows * cols * sizeof(float**));
    float **res_2  = (float**)malloc(rows * cols * sizeof(float**));
    float *f_1     = (float*)malloc(rows*cols * D * sizeof(float*));
    float *res_1   = (float*)malloc(rows*cols * D * sizeof(float*));
    uint *image   = (uint*)malloc(rows*cols * sizeof(uint*));

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
    uint *device_image;
    cudaMalloc((void**)(&d_1), rows*cols * D * sizeof(float));
    cudaMalloc((void**)(&d_res_1), rows*cols * D * sizeof(float));
    cudaMalloc((void**)(&device_image), rows*cols * sizeof(uint));

//    for (int i = 0; i < rows*cols * D; i++)// ini the data 3D
//    {
//        f_1[i] = 0;
//    }
    int index=0;
    int index_image=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            image[index_image++]=left_image.at<uchar>(i,j);
            for(int k=0;k<D;k++){
                f_1[index++]=cost_disp[i][j][k];
            }
        }
    }
    cudaMemcpy(device_image,image,rows*cols*sizeof(uint),cudaMemcpyHostToDevice);
    cudaError_t error0=cudaMemcpy(d_1, f_1, rows*cols * D * sizeof(float), cudaMemcpyHostToDevice);
    //cout<<"0: "<<error0<<endl;
    cudaError_t error1=cudaMemcpy(d_res_1, res_1, rows*cols * D * sizeof(float), cudaMemcpyHostToDevice);
    //cout<<"1: "<<error1<<endl;
    for (int i = 0; i < rows*cols ; i++){
        f_2[i] = d_1 + i * D;
        res_2[i] = d_res_1 + i * D;
    }
    cudaError_t error2=cudaMemcpy(d_2, f_2, rows*cols * sizeof(float**), cudaMemcpyHostToDevice);
    //cout<<"2: "<<error2<<endl;
    cudaError_t error3=cudaMemcpy(d_res_2, res_2, rows*cols * sizeof(float**), cudaMemcpyHostToDevice);
    //cout<<"3: "<<error3<<endl;
    for (int i = 0; i < rows; i++){
        f_3[i] = d_2 + cols * i;
        res_3[i] = d_res_2 + cols * i;
    }
    cudaError_t error4=cudaMemcpy(d_3, f_3, rows * sizeof(float***), cudaMemcpyHostToDevice);
    //cout<<"4: "<<error4<<endl;
    cudaError_t error5=cudaMemcpy(d_res_3, res_3, rows * sizeof(float***), cudaMemcpyHostToDevice);
    //cout<<"5: "<<error5<<endl;
    cudaError_t error6=cudaDeviceSynchronize();
    //cout<<"6: "<<error6<<endl;
    //////////////////////////// end /////////////////////////
    /////////////////////////// four path aggregation ////////
    dim3 grid(rows);
    dim3 block(D);

    // aggregation from left to right
    auto start=system_clock::now();
    AggreKeral_left<<<grid,block>>>(d_3,d_res_3,device_image,cols,D);
    cudaDeviceSynchronize();
    //aggregation from right to left
    AggreKeral_right<<<grid,block>>>(d_res_1,d_1,device_image,rows,cols,D);
    cudaDeviceSynchronize();
    // aggregation from top to down
    dim3 grid_td(cols);
    dim3 block_td(D);
    Aggrekeral_top<<<grid_td,block_td>>>(d_1,d_res_1,device_image,rows,cols,D);
    cudaDeviceSynchronize();
    Aggrekeral_down<<<grid_td,block_td>>>(d_res_1,d_1,device_image,rows,cols,D);
    cudaDeviceSynchronize();
    cudaError_t cudaStatus=cudaGetLastError();
    if(cudaStatus!=cudaSuccess){
        fprintf(stderr,"aggregation failed:&s\n",cudaGetErrorString(cudaStatus));
    }
    cudaError_t error7=cudaDeviceSynchronize(); // synchronization
    ////////////////////////// end //////////////////////////
    /////////// download the data from device ///////////////
    //cudaError_t error11=cudaMemcpy(f_1,d_res_1,rows*cols * D * sizeof(float),cudaMemcpyDeviceToHost);
    cudaError_t error11=cudaMemcpy(f_1,d_1,rows*cols * D * sizeof(float),cudaMemcpyDeviceToHost);
    //cout<<"11: "<<error11<<endl;
    index=0;
    for(int i=0;i<rows;i++){
        for(int j=0;j<cols;j++){
            for(int k=0;k<D;k++){
                cost_disp[i][j][k]=f_1[index++];
            }
        }
    }
    auto last=system_clock::now();
    auto duration=duration_cast<microseconds>(last-start);
    cout<<"aggregation cost total time:  "
        <<double(duration.count())*microseconds::period::num/microseconds::period::den<<endl;
    /////////// end  ////////////
    /////////// free  ///////////
    free(f_3);
    free(f_2);
    free(f_1);
    cudaFree(d_3);
    cudaFree(d_2);
    cudaFree(d_1);
    free(res_3);
    free(res_2);
    free(res_1);
    free(image);
    cudaFree(d_res_3);
    cudaFree(d_res_2);
    cudaFree(d_res_1);
    cudaFree(device_image);
    cout<<"cuda aggregation done..."<<endl;
    ////////////////////
    return 0;
}