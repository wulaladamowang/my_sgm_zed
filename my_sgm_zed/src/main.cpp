#include "head.hpp"

void get_remap(const std::string in, const std::string out, const int width, const int height, std::vector<cv::Mat>& maps){
    //std::string in = "/home/wang/code/c++Code/my_sgm_zed_server/canshu/intrinsics.yml";
    cv::FileStorage fs(in, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file IN\n");
        return;
    }
    cv::Mat M1, D1, M2, D2;
    fs["M1"] >> M1;
    fs["D1"] >> D1;
    fs["M2"] >> M2;
    fs["D2"] >> D2;
    //std::string out = "/home/wang/code/c++Code/my_sgm_zed_server/canshu/extrinsics.yml";
    fs.open(out, cv::FileStorage::READ);
    if(!fs.isOpened())
    {
        printf("Failed to open file OUT\n");
        return;
    }
    cv::Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;
    cv::Mat Q;
    cv::Size img_size = cv::Size(width, height);
    cv::stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, cv::CALIB_ZERO_DISPARITY, -1, img_size, nullptr, nullptr );
    cv::Mat map11, map12, map21, map22;//校正参数
    cv::initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    cv::initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);
    maps = {map11, map12, map21, map22};
}

int main(int argc, char** argv){

// zed相机的初始化
	sl::Camera zed;
	sl::InitParameters initParameters;
	initParameters.camera_resolution = sl::RESOLUTION::HD1080;
    if (argc >= 5) initParameters.input.setFromSVOFile(argv[4]);
	sl::ERROR_CODE err = zed.open(initParameters);
	if (err != sl::ERROR_CODE::SUCCESS) {
		std::cout << toString(err) << std::endl;
		zed.close();
		return 1;
	}
    sl::RuntimeParameters runtime_parameters;
    runtime_parameters.sensing_mode = sl::SENSING_MODE::LAST;

//图像变量的获取及初始化

    const int width = static_cast<int>(zed.getCameraInformation().camera_resolution.width);
	const int height = static_cast<int>(zed.getCameraInformation().camera_resolution.height);
    sl::Mat zed_image_l(zed.getCameraInformation().camera_resolution, sl::MAT_TYPE::U8_C1);
	sl::Mat zed_image_r(zed.getCameraInformation().camera_resolution, sl::MAT_TYPE::U8_C1);//相机原格式获得图像
    cv::Mat img_zed_left = slMat2cvMat(zed_image_l);
    cv::Mat img_zed_right = slMat2cvMat(zed_image_r);//sl::Mat 到opencv 格式图像的转换
    cv::Mat img_zed_left_remap(img_zed_left.size(), img_zed_left.type());// 存储校正畸变后的图像
    cv::Mat img_zed_right_remap(img_zed_right.size(), img_zed_right.type());

// 相机内外参数的读取, 注意相机内外参数要与使用的相机型号或拍摄视频的相机型号相一致
    std::string in = "/home/wang/code/c++Code/my_sgm_zed_server/canshu/intrinsics.yml";
    std::string out = "/home/wang/code/c++Code/my_sgm_zed_server/canshu/extrinsics.yml";
    cv::Size img_size = cv::Size(width, height);
    std::vector<cv::Mat> map;// 映射矩阵容器
    get_remap(in, out, width, height, std::ref(map));

//立体匹配参数设定
    const float scale = argc >= 2 ? atof(argv[1]) : 0.5;//图像缩放比例,默认为变为原来的0.5倍，图像大小对sgm算法影响较大
    const int disp_size = (argc >= 3) ? std::stoi(argv[2]) : 128;//默认的disparity size, 可选择64,128,256
    const bool subpixel = (argc >= 4) ? std::stoi(argv[3]) != 0 : true;//是否使用subpixel

    cv::Size siz_scale = cv::Size(width*scale, height*scale);// 对于原图进行缩放尺寸，缩放后的尺寸大小
    cv::Mat disparity_8u(siz_scale, CV_8U), disparity_32f(siz_scale, CV_32F), disparity_mask(siz_scale, CV_8U);// 8u转化便于显示的灰度图, 32f实际视差图,带小数点, disparity_mask保留mask区域的视差
    
    const int input_depth = img_zed_left.type() == CV_8U ? 8 : 16;
    const int output_depth = 16;
    const sgm::StereoSGM::Parameters params{10, 120, 0.95f, subpixel};
    sgm::StereoSGM sgm(width*scale, height*scale, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA, params);

    bool run = true;
    struct img_time zed_img_time(height, width, img_zed_left.type());
    struct roi_time roi_mask(height, width);
    struct disparity_time disparity_and_time(height*scale, width*scale, CV_16S);

    std::thread GetImg(getImage, std::ref(zed), std::ref(runtime_parameters), std::ref(map), std::ref(zed_img_time), std::ref(run));
    //std::thread GetRoi(getMaskRoi, std::ref(zed_img_time), std::ref(roi_mask), std::ref(run));
    //std::thread GetDisparity(getDisparity, std::ref(sgm), std::ref(zed_img_time), std::ref(disparity_and_time), std::ref(run));

    while(1){
        cv::imshow("Image", zed_img_time.img_left);
        std::cout << zed_img_time.time << std::endl;
        const char key = cv::waitKey(1);
        if (key == 27)
            break;
    }
    run = false;
    GetImg.join();
    //GetRoi.join();
    //GetDisparity.join();
    //zed.close();// 增加此条语句会造成 段错误， 已在自带zed例子上进行验证，程序自带bug
    std::cout << "finish ---------------- " << std::endl;
    return 0;
}

