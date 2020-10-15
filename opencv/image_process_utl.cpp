//
// Created by maxwell on 9/8/20.
//

#include "image_process_utl.h"
///
///排序算法-冒泡排序(改进后)
///
void bublle_sort(std::vector<int> &arr){
    bool flag=true;
    for (int i = 0; i < arr.size() - 1; ++i){
        while (flag){
            flag = false;
            for (int j = 0; j < arr.size() - 1 - i; ++j){
                if (arr[j]>arr[j + 1]){
                    int tmp = arr[j];
                    arr[j] = arr[j + 1];
                    arr[j + 1] = tmp;
                    flag = true;
                }
            }
        }
    }
}
//中值滤波
void MedianFilter(const float* in, float* out, const int32_t& width, const int32_t& height,const int32_t wnd_size){
    const int32_t radius = wnd_size / 2;
    const int32_t size = wnd_size * wnd_size;
    // 存储局部窗口内的数据
    std::vector<double> wnd_data;
    wnd_data.reserve(size);
    for (int32_t i = 0; i < height; i++) {
        for (int32_t j = 0; j < width; j++) {
            wnd_data.clear();
            // 获取局部窗口数据
            for (int32_t r = -radius; r <= radius; r++) {
                for (int32_t c = -radius; c <= radius; c++) {
                    const int32_t row = i + r;
                    const int32_t col = j + c;
                    if (row >= 0 && row < height && col >= 0 && col < width) {
                        wnd_data.push_back(in[row * width + col]);
                    }
                }
            }
            // 排序
            std::sort(wnd_data.begin(), wnd_data.end());
            // 取中值
            out[i * width + j] = wnd_data[wnd_data.size() / 2];
        }
    }
}
float calculate_corss_correlation(float *s1, float *s2,int n)
{
    double delta   = 0.0001f;
    double sum_s12 = 0.0;
    double sum_s1  = 0.0;
    double sum_s2  = 0.0;
    double sum_s1s1 = 0.0; //s1^2
    double sum_s2s2 = 0.0; //s2^2
    double pxy = 0.0;
    double temp1 = 0.0;
    double temp2 = 0.0;
    if( s1==NULL || s2==NULL || n<=0)
        return -10;
    for(int i=0;i<n;i++)
    {
        sum_s12  += s1[i]*s2[i];
        sum_s1   += s1[i];
        sum_s2   += s2[i];
        sum_s1s1 += s1[i]*s1[i];
        sum_s2s2 += s2[i]*s2[i];
        // printf("S1: %f,S2: %f\n",s1[i],s2[i]);
    }
    temp1 = n*sum_s1s1-sum_s1*sum_s1;
    temp2 = n*sum_s2s2-sum_s2*sum_s2;
    if( (temp1>-delta && temp1<delta) ||
        (temp2>-delta && temp2<delta) ||
        (temp1*temp2<=0) )
    {
        return -10;
    }
    pxy = (n*sum_s12-sum_s1*sum_s2)/sqrt(temp1*temp2);
   // printf("process...: sqrt: %f ,temp1: %f ,temp2: %f ---",sqrt(temp1*temp2),temp1,temp2);
   // printf("pre : %f pxy %f\n",n*sum_s12-sum_s1*sum_s2,pxy);
    return (float_t)pxy;
}
