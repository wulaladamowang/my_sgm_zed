#ifndef __GET_DISPARITY_H__
#define __GET_DISPARITY_H__

#include "little_tips.hpp"
#include "my_var.hpp"
#include "libsgm.h"

int get_disparity(sgm::StereoSGM& sgm, const cv::Mat& img_left, const cv::Mat& img_right, cv::Mat& disparity);
void getDisparity(sgm::StereoSGM& sgm, const struct img_time& img_zed, const cv::Size& siz_scale, struct disparity_time& disparity_and_time, bool& run);

#endif