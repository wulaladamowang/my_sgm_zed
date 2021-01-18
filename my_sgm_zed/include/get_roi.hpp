#ifndef __GET_ROI_H__
#define __GET_ROI_H__

#include <iostream>
#include <opencv2/aruco.hpp>
#include "my_var.hpp"

int relativeDis(cv::Vec4f line_para, std::vector<cv::Point2f> point);
void get_roi(cv::Mat& image, cv::Mat& mask, bool& has_roi, std::vector<int>& rect_roi) ;
void getMaskRoi(struct img_time& img_zed, struct roi_time& roi_mask, bool& run);

#endif