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
void MedianFilter(cv::Mat& src, cv::Mat& dst, cv::Size wsize){
    //图像边界扩充
    if (wsize.width % 2 == 0 || wsize.height % 2 == 0){
        fprintf(stderr, "Please enter odd size!");
        exit(-1);
    }
    int hh = (wsize.height - 1) / 2;
    int hw = (wsize.width - 1) / 2;
    cv::Mat Newsrc;
    cv::copyMakeBorder(src, Newsrc, hh, hh, hw, hw, cv::BORDER_REFLECT_101);//以边缘为轴，对称
    dst = cv::Mat::zeros(src.rows, src.cols, src.type());

    //中值滤波
    for (int i = hh; i < src.rows + hh; ++i){
        uchar* ptrdst = dst.ptr(i - hh);
        for (int j = hw; j < src.cols + hw; ++j){
            std::vector<int> pix;
            for (int r = i - hh; r <= i + hh; ++r){
                const uchar* ptrsrc = Newsrc.ptr(r);
                for (int c = j - hw; c <= j + hw; ++c){
                    pix.push_back(ptrsrc[c]);
                }
            }
            bublle_sort(pix);//冒泡排序
            ptrdst[j - hw] = pix[(wsize.area() - 1) / 2];//将中值映射到输出图像
        }
    }
}
