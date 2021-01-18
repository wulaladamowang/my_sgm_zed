#ifndef __MY_VAR_H__
#define __MY_VAR_H__

#include "little_tips.hpp"
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

// 存储图像及其更新的时间
struct img_time{
    cv::Mat img_left;
    cv::Mat img_right;
    long long time;
    img_time(int rows, int cols, int type):
        img_left(cv::Mat(rows, cols, type)),
        img_right(cv::Mat(rows, cols, type)){
            time = getCurrentTime();
        };
};

// 存储目标检测结果及其时间
struct roi_time{
    cv::Mat mask;
    std::vector<int> rect_roi;
    bool is_detected_mask;
    long long time;
    roi_time(int rows, int cols):mask(cv::Mat(rows, cols, CV_8UC1)){
        is_detected_mask = false;
        rect_roi.reserve(4);
        time = getCurrentTime();
    }
};

// 存储视差图及其获得时间
struct disparity_time{
    cv::Mat disparity;
    long long time;
    disparity_time(int rows, int cols, int type):
        disparity(cv::Mat(rows, cols, type)){
            time = getCurrentTime();
        }
};

struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};

#endif