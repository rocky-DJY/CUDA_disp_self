//created by biubiu on 20200803
#include "disp_main.h"
#include "computeXYZ.h"
int DispMain() {
	// Create a ZED camera object
	sl::Camera zed;
	// Set configuration parameters
	InitParameters init_params;
	init_params.camera_resolution = RESOLUTION::HD1080;
	init_params.depth_mode        = DEPTH_MODE::ULTRA;
	init_params.coordinate_units  = UNIT::MILLIMETER;
	//zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, VIDEO_SETTINGS_VALUE_AUTO);

	//init_params.depth_maximum_distance = 1500;
	//if (argc > 1) init_params.input.setFromSVOFile(argv[1]);
	// Open the camera
	ERROR_CODE err = zed.open(init_params);
	if (err != ERROR_CODE::SUCCESS) { // 打开失败则退出
		printf("%s\n", toString(err).c_str());
		zed.close();
		return 1; // Quit if an error occurred
	}
	// zed open 之后才能设置参数
//	zed.setCameraSettings(VIDEO_SETTINGS::GAIN, 50);      // 50
//	zed.setCameraSettings(VIDEO_SETTINGS::BRIGHTNESS, 4); // 4
//	zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, 50);  // 50
	zed.setCameraSettings(VIDEO_SETTINGS::GAIN,VIDEO_SETTINGS_VALUE_AUTO);
	zed.setCameraSettings(VIDEO_SETTINGS::BRIGHTNESS,VIDEO_SETTINGS_VALUE_AUTO);
	zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE,VIDEO_SETTINGS_VALUE_AUTO);
	zed.setCameraSettings(VIDEO_SETTINGS::WHITEBALANCE_AUTO);
	// Set runtime parameters after opening the camera
	RuntimeParameters runtime_parameters;
	// runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;
	runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;

	// Prepare new image size to retrieve half-resolution images
	sl::Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;
	int new_width  = image_size.width  / 1;
	int new_height = image_size.height / 1;
	sl::Resolution new_image_size(new_width, new_height);
	// 左相机的zed校正图片  sl空间和opencv空间
	sl::Mat image_zed_undistort_left(new_width, new_height,MAT_TYPE::U8_C4);
	cv::Mat image_ocv_undistort_left = slMat2cvMat(image_zed_undistort_left);
	// 右相机的zed校正图片  sl空间和opencv空间
	sl::Mat image_zed_undistort_right(new_width, new_height, MAT_TYPE::U8_C4);
	cv::Mat image_ocv_undistort_right= slMat2cvMat(image_zed_undistort_right);
    cv::Mat LEFT_image,RIGHT_image;
    //********** 相机参数 *********//
    cv::Mat leftIntrinsic=(cv::Mat_<double>(3,3)<<
            1399.08 ,    0    , 1120.72,
               0    , 1399.08 , 688.597,
               0    ,    0    , 1
            );
    cv::Mat rightIntrinsic=(cv::Mat_<double>(3,3)<<
            1400.84 ,    0    , 1114.24,
               0    , 1400.84 , 712.094,
               0    ,    0    , 1
            );
    // cv::Mat r_vector=(cv::Mat_<double>(3,1)<<0.0043,-0.0204,-0.0003);
    cv::Mat rightRotation=(cv::Mat_<double>(3,3)<<
            1,0,0,
            0,1,0,
            0,0,1
            );
    // cv::Rodrigues(r_vector,rightRotation);
    cv::Mat rightTranslation=(cv::Mat_<double>(3,1)<<-120.000,0,0);

    //********* end *************//
	char key = ' ';
	int counter=0;
	while (key!='q') {
	    if(counter<20&zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS){
            // Retrieve the left image, depth image in half-resolution
            zed.retrieveImage(image_zed_undistort_left, VIEW::LEFT, MEM::CPU, new_image_size);
            zed.retrieveImage(image_zed_undistort_right, VIEW::RIGHT, MEM::CPU, new_image_size);
	        counter++;
	    }
	    else{
            if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {
                // Retrieve the left image, depth image in half-resolution
                zed.retrieveImage(image_zed_undistort_left,  VIEW::LEFT,  MEM::CPU, new_image_size);
                zed.retrieveImage(image_zed_undistort_right, VIEW::RIGHT, MEM::CPU, new_image_size);
                cv::cvtColor(image_ocv_undistort_left,LEFT_image,CV_BGRA2BGR);
                // cv::cvtColor(image_ocv_undistort_left,image_ocv_undistort_left,CV_BGR2HSV);
                cv::cvtColor(image_ocv_undistort_right,RIGHT_image,CV_BGRA2BGR);
                // cv::cvtColor(image_ocv_undistort_right,image_ocv_undistort_right,CV_BGR2HSV);
                // cout << point_cloud.getWidth() << "," << point_cloud.getHeight() << endl;
                // Display image and depth using cv:Mat which share sl:Mat data
                cv::namedWindow("Image_left",0);
                cv::resizeWindow("Image_left",(int)1920/3,(int)1080/3);
                cv::imshow("Image_left", LEFT_image);
                cv::imwrite("LeftImage.bmp",LEFT_image);   // write

                cv::namedWindow("Image_right",0);
                cv::resizeWindow("Image_right",(int)1920/3,(int)1080/3);
                cv::imshow("Image_right",RIGHT_image);
                cv::imwrite("RightImage.bmp",RIGHT_image);   // write
                cvWaitKey(1500);
                // 视差计算  create obj to compute disp  set win_size
                dispart_estimate disp_obj(5,5);  // census win_size w,h
                cv::Mat disp_Image;
                cv::Mat disp_map;
                disp_map=disp_obj.compute_disp(LEFT_image,RIGHT_image,disp_Image);  //返回视差真实值
                // 点云坐标计算 实例
                computeXYZ* xyz_obj=new computeXYZ(LEFT_image,leftIntrinsic,rightIntrinsic,rightRotation,rightTranslation);
                xyz_obj->Disptopixelcorr(disp_map);
                xyz_obj->allcompute();
                xyz_obj->Save_xyz_txt("xyz.txt");
                delete xyz_obj;
                cout<<"save xyz done!"<<endl;
//                cv::Mat image_left=cv::imread("/home/maxwell/Downloads/Bicycle1-perfect/im0.png");               //  1
//                cv::Mat image_right=cv::imread("/home/maxwell/Downloads/Bicycle1-perfect/im1.png");
//                int scale=2;
//                cv::resize(image_left,image_left,cv::Size((int)image_left.cols/scale,(int)image_left.rows/scale));
//                cv::resize(image_right,image_right,cv::Size((int)image_right.cols/scale,(int)image_right.rows/scale));
//                disp_obj.compute_disp(image_left,image_right,disp_Image);

                cv::namedWindow("Disp_image",0);
                cv::resizeWindow("Disp_image",(int)disp_Image.cols/3,(int)disp_Image.rows/3);
                cv::imshow("Disp_image",disp_Image);
                cv::waitKey(0);
                cv::destroyWindow("Disp_image");
                cv::imwrite("disp_unagg.bmp",disp_Image);
                cout<<"disp done enter the key order... q: exit n: continue"<<endl;
                // Handle key event
                while(true){
                    key = cv::waitKey(10);
                    if(key=='q')  // exit
                        break;
                    if(key=='n')  // continue;
                        break;
                }
            }
		}
	}
	zed.close();
	return 1;
}
cv::Mat slMat2cvMat(Mat& input) {
	// Mapping between MAT_TYPE and CV_TYPE
	int cv_type = -1;
	switch (input.getDataType()) {
	case MAT_TYPE::F32_C1: cv_type = CV_32FC1; break;
	case MAT_TYPE::F32_C2: cv_type = CV_32FC2; break;
	case MAT_TYPE::F32_C3: cv_type = CV_32FC3; break;
	case MAT_TYPE::F32_C4: cv_type = CV_32FC4; break;
	case MAT_TYPE::U8_C1: cv_type = CV_8UC1; break;
	case MAT_TYPE::U8_C2: cv_type = CV_8UC2; break;
	case MAT_TYPE::U8_C3: cv_type = CV_8UC3; break;
	case MAT_TYPE::U8_C4: cv_type = CV_8UC4; break;
	default: break;
	}
	// Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
	// cv::Mat and sl::Mat will share a single memory structure
	return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM::CPU));
}