//created by biubiu on 20200803
#include "disp_main.h"

int DispMain() {
	// Create a ZED camera object
	Camera zed;
	// Set configuration parameters
	InitParameters init_params;
	init_params.camera_resolution = RESOLUTION::HD1080;
	init_params.depth_mode = DEPTH_MODE::ULTRA;
	init_params.coordinate_units = UNIT::MILLIMETER;
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
	zed.setCameraSettings(VIDEO_SETTINGS::GAIN, 50);
	zed.setCameraSettings(VIDEO_SETTINGS::BRIGHTNESS, 4);
	zed.setCameraSettings(VIDEO_SETTINGS::EXPOSURE, 50);
	// Set runtime parameters after opening the camera
	RuntimeParameters runtime_parameters;
	runtime_parameters.sensing_mode = SENSING_MODE::STANDARD;
	
	// Prepare new image size to retrieve half-resolution images
	Resolution image_size = zed.getCameraInformation().camera_configuration.resolution;
	int new_width  = image_size.width / 1;
	int new_height = image_size.height / 1;
	Resolution new_image_size(new_width, new_height);
	// 左相机的zed校正图片  sl空间和opencv空间
	Mat image_zed_undistort_left(new_width, new_height, MAT_TYPE::U8_C4);
	cv::Mat image_ocv_undistort_left = slMat2cvMat(image_zed_undistort_left);
	// 右相机的zed校正图片  sl空间和opencv空间
	Mat image_zed_undistort_right(new_width, new_height, MAT_TYPE::U8_C4);
	cv::Mat image_ocv_undistort_right= slMat2cvMat(image_zed_undistort_right);

	char key = ' ';
	int counter=0;
	while (key!='q') {
	    if(counter<30&zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS){
            // Retrieve the left image, depth image in half-resolution
            zed.retrieveImage(image_zed_undistort_left, VIEW::LEFT, MEM::CPU, new_image_size);
            zed.retrieveImage(image_zed_undistort_right, VIEW::RIGHT, MEM::CPU, new_image_size);
	        counter++;
	    }
	    else{
            if (zed.grab(runtime_parameters) == ERROR_CODE::SUCCESS) {
                // Retrieve the left image, depth image in half-resolution
                zed.retrieveImage(image_zed_undistort_left, VIEW::LEFT, MEM::CPU, new_image_size);
                zed.retrieveImage(image_zed_undistort_right, VIEW::RIGHT, MEM::CPU, new_image_size);
                //cout << point_cloud.getWidth() << "," << point_cloud.getHeight() << endl;
                // Display image and depth using cv:Mat which share sl:Mat data
                cv::namedWindow("Image_left",0);
                cv::resizeWindow("Image_left",(int)1920/3,(int)1080/3);
                cv::imshow("Image_left", image_ocv_undistort_left);
                cv::namedWindow("Image_right",0);
                cv::resizeWindow("Image_right",(int)1920/3,(int)1080/3);
                cv::imshow("Image_right", image_ocv_undistort_right);
                cvWaitKey(1500);
                // new census obj 需要手动释放对象的内存
                census *CT_obj_left  = new census(0);
                census *CT_obj_right = new census(1);
                int winsize_x = 5;
                int winsize_y = 5;
                cv::Mat census_image_left, census_image_right; // census 变换结果 使用64F datatype
                CT_obj_left ->census_transform(image_ocv_undistort_left,  census_image_left, winsize_x,winsize_y);
                CT_obj_right->census_transform(image_ocv_undistort_right, census_image_right,winsize_x,winsize_y);

                //cv::imshow("census_transfrom_L", census_image_left);
                //cv::imshow("census_transfrom_R", census_image_right);
                // 手动释放内存
                CT_obj_left ->~census();
                CT_obj_right->~census();

                // 视差计算
                dispart_estimate disp_obj(image_ocv_undistort_left,image_ocv_undistort_right);
                cv::Mat disp_Image;
                disp_obj.compute_disp(census_image_left, census_image_right,disp_Image);
                cv::namedWindow("Disp_image",0);
                cv::resizeWindow("Disp_image",(int)1920/3,(int)1080/3);
                cv::imshow("Disp_image",disp_Image);
                cv::waitKey(0);
                cv::destroyWindow("Disp_image");
                cout<<"disp done enter the key order..."<<endl;
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