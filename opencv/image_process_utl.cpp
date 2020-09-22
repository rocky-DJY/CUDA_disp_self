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
    std::vector<float> wnd_data;
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
