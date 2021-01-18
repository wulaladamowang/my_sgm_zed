#include "get_disparity.hpp"

int get_disparity(sgm::StereoSGM& sgm, const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& disparity){
    static const int width = img_left.cols;
    static const int height = img_left.rows;
    
    static const int input_depth = img_left.type() == CV_8U ? 8 : 16;
    static const int input_bytes = input_depth * width * height / 8;
    static const int output_depth = 16;
    static const int output_bytes = output_depth * width * height / 8;


    static device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);

    cudaMemcpy(d_I1.data, img_left.data, input_bytes, cudaMemcpyHostToDevice);//input_bytes : nBytes
	cudaMemcpy(d_I2.data, img_right.data, input_bytes, cudaMemcpyHostToDevice);
    sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
    cudaDeviceSynchronize();
    cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
    return 0;
}

void getDisparity(sgm::StereoSGM& sgm, const struct img_time& img_zed, const cv::Size& siz_scale, struct disparity_time& disparity_and_time, bool& run){
    cv::Mat img_left_scale(siz_scale, img_zed.img_left.type());
    cv::Mat img_right_scale(siz_scale, img_zed.img_left.type());
    while (run){
        cv::resize(img_zed.img_left, img_left_scale, siz_scale);
        cv::resize(img_zed.img_right, img_right_scale, siz_scale);
        get_disparity(std::ref(sgm), std::ref(img_left_scale), std::ref(img_right_scale), std::ref(disparity_and_time.disparity));
        disparity_and_time.time = getCurrentTime();
    }
        
}