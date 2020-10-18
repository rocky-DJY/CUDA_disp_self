//
// Created by maxwell on 10/16/20.
//
#include "computeXYZ.h"
using namespace std;
//using namespace cv;
//*****构造传参*****//
computeXYZ::computeXYZ(cv::Mat image_left,cv::Mat leftIntrinsic,cv::Mat rightIntrinsic, cv::Mat rightRotation, cv::Mat rightTranslation) {
    this->image_left=image_left;
    this->leftIntrinsic=leftIntrinsic;
    this->rightIntrinsic=rightIntrinsic;
    this->rightRotation =rightRotation;
    this->rightTranslation=rightTranslation;
    //  世界坐标系到左相机坐标系的旋转矩阵
    this->leftRotation    = (cv::Mat_<double>(3, 3) << 1, 0, 0,0,1,0,0,0,1);
    //  世界坐标系到左相机坐标系的平移矩阵
    this->leftTranslation = (cv::Mat_<double>(3, 1) << 0 , 0,  0);
}
computeXYZ::~computeXYZ() {
    leftIntrinsic.release();
    leftTranslation.release();
    leftRotation.release();
    rightIntrinsic.release();
    rightRotation.release();
    rightTranslation.release();
    image_left.release();
}
///////////
cv::Mat computeXYZ::compute(cv::Point2d uvLeft, cv::Point2d uvRight)
{
    cv::Mat mLeftRotation = cv::Mat_<double>(3, 3);
    mLeftRotation = leftRotation;
    cv::Mat mLeftTranslation = cv::Mat_<double>(3, 1);
    mLeftTranslation = leftTranslation;
    cv::Mat mLeftRT = cv::Mat_<double>(3, 4);//×óÏà»úRTŸØÕó
    hconcat(mLeftRotation, mLeftTranslation, mLeftRT);
    cv::Mat mLeftIntrinsic = cv::Mat_<double>(3, 3);
    mLeftIntrinsic=leftIntrinsic;   // Mat(3, 3, CV_32F, leftIntrinsic);
    cv::Mat mLeftM = mLeftIntrinsic * mLeftRT;

    cv::Mat mRightRotation = rightRotation;       // Mat(3, 3, CV_64F, rightRotation);    64F=double Type
    cv::Mat mRightTranslation = rightTranslation; // Mat(3, 1, CV_64F, rightTranslation);
    cv::Mat mRightRT = cv::Mat_<double>(3, 4);        //ÓÒÏà»úMŸØÕó
    hconcat(mRightRotation, mRightTranslation, mRightRT);
    cv::Mat mRightIntrinsic=cv::Mat_<double>(3,3);
    mRightIntrinsic = rightIntrinsic;     // Mat(3, 3, CV_32F, rightIntrinsic);
    cv::Mat mRightM = mRightIntrinsic * mRightRT;

//    cout<<"left 内参： \n"<<mLeftIntrinsic<<endl;
//    cout<<"right 内参: \n"<<mRightIntrinsic<<endl;
//    cout<<"R:\n"<<mRightRotation<<endl;
//    cout<<"T:\n"<<mRightTranslation<<endl;

    cv::Mat A = cv::Mat_<double>(4, 3);
    A.at<double>(0, 0) = uvLeft.x * mLeftM.at<double>(2, 0) - mLeftM.at<double>(0, 0);
    A.at<double>(0, 1) = uvLeft.x * mLeftM.at<double>(2, 1) - mLeftM.at<double>(0, 1);
    A.at<double>(0, 2) = uvLeft.x * mLeftM.at<double>(2, 2) - mLeftM.at<double>(0, 2);

    A.at<double>(1, 0) = uvLeft.y * mLeftM.at<double>(2, 0) - mLeftM.at<double>(1, 0);
    A.at<double>(1, 1) = uvLeft.y * mLeftM.at<double>(2, 1) - mLeftM.at<double>(1, 1);
    A.at<double>(1, 2) = uvLeft.y * mLeftM.at<double>(2, 2) - mLeftM.at<double>(1, 2);

    A.at<double>(2, 0) = uvRight.x * mRightM.at<double>(2, 0) - mRightM.at<double>(0, 0);
    A.at<double>(2, 1) = uvRight.x * mRightM.at<double>(2, 1) - mRightM.at<double>(0, 1);
    A.at<double>(2, 2) = uvRight.x * mRightM.at<double>(2, 2) - mRightM.at<double>(0, 2);

    A.at<double>(3, 0) = uvRight.y * mRightM.at<double>(2, 0) - mRightM.at<double>(1, 0);
    A.at<double>(3, 1) = uvRight.y * mRightM.at<double>(2, 1) - mRightM.at<double>(1, 1);
    A.at<double>(3, 2) = uvRight.y * mRightM.at<double>(2, 2) - mRightM.at<double>(1, 2);

    //×îÐ¡¶þ³Ë·šBŸØÕó
    cv::Mat B = cv::Mat_<double>(4, 1);
    B.at<double>(0, 0) = mLeftM.at<double>(0, 3) - uvLeft.x * mLeftM.at<double>(2, 3);
    B.at<double>(1, 0) = mLeftM.at<double>(1, 3) - uvLeft.y * mLeftM.at<double>(2, 3);
    B.at<double>(2, 0) = mRightM.at<double>(0, 3) - uvRight.x * mRightM.at<double>(2, 3);
    B.at<double>(3, 0) = mRightM.at<double>(1, 3) - uvRight.y * mRightM.at<double>(2, 3);

    cv::Mat XYZ = cv::Mat_<double>(3, 1);
    //²ÉÓÃSVD×îÐ¡¶þ³Ë·šÇóœâXYZ
    cv::solve(A, B, XYZ, cv::DECOMP_SVD);
    return XYZ;
}
void computeXYZ::Disptopixelcorr(cv::Mat DispMap){
    // 去除边框
    int boundry_size=100;
    int cols_size=DispMap.cols-boundry_size;
    int rows_size=DispMap.cols-boundry_size;
    // 左视图为base
    for(int index_row=boundry_size;index_row<rows_size-boundry_size;index_row++){
        // 图像行指针
        const uchar* row_ptr=DispMap.ptr<uchar>(index_row);
        for(int index_col=boundry_size;index_col<cols_size-boundry_size;index_col++){
            cv::Point2d left_temp,right_temp;
            left_temp.x=index_col;
            left_temp.y=index_row;
            // 列值索引取出视差
            right_temp.x=(double)(index_col-row_ptr[index_col]);
            right_temp.y=index_row;
            pixels_left.push_back(left_temp);
            pixels_right.push_back(right_temp);
            // cout<<"left: "<<left_temp<<"  "<<"right: "<<right_temp<<endl;
        }
    }
}
void computeXYZ::allcompute(){
    for(int i=0;i<pixels_left.size();i++){
        cv::Point3d xyztemp;
        cv::Point3d rgbtemp;
        cv::Mat XYZ_pixel=compute(pixels_left[i],pixels_right[i]);
        xyztemp.x=XYZ_pixel.at<double>(0);
        xyztemp.y=XYZ_pixel.at<double>(1);
        xyztemp.z=XYZ_pixel.at<double>(2);
        rgbtemp.x=image_left.at<cv::Vec3b>(pixels_left[i].y,pixels_left[i].x)[2];
        rgbtemp.y=image_left.at<cv::Vec3b>(pixels_left[i].y,pixels_left[i].x)[1];
        rgbtemp.z=image_left.at<cv::Vec3b>(pixels_left[i].y,pixels_left[i].x)[0];
        world_xyz.push_back(xyztemp);
        world_rgb.push_back(rgbtemp);
    }
    return;
}
// 保存到本地文件夹
void computeXYZ::Save_xyz_txt(const char* filename) {
    FILE* fp=fopen(filename,"w");
    int len=world_xyz.size();
    for(int i=0;i<len;i++){
        fprintf(fp,"%lf;%lf;%lf;%lf;%lf;%lf\n",world_xyz[i].x,world_xyz[i].y,world_xyz[i].z,world_rgb[i].x,world_rgb[i].y,world_rgb[i].z);
    }
    fclose(fp);
}

