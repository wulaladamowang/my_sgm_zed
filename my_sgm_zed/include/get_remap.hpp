#ifndef __GET_REMAP_H__
#define __GET_REMAP_H__

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

void get_remap(const std::string in, const std::string out, const int width, const int height, std::vector<cv::Mat>& maps);

#endif