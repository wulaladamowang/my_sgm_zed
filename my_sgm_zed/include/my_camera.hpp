/*
 * @Author: your name
 * @Date: 2020-12-12 10:32:45
 * @LastEditTime: 2020-12-12 16:03:04
 * @LastEditors: Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /sgm_code/include/init_camera.h
 */
#ifndef __INIT_CAMERA_H__
#define __INIT_CAMERA_H__

#include <sl/Camera.hpp>
#include "my_var.hpp"
#include "little_tips.hpp"

int init_camera_parameters(int nb_zeds, std::vector<sl::Camera>& zeds);
void zed_acquisition(sl::Camera& zed, cv::Mat& img_left, cv::Mat& img_right, std::vector<cv::Mat>& map, bool& run, long long& ts);
void getImage(sl::Camera& zed, const sl::RuntimeParameters& runtime_parameters, const std::vector<cv::Mat>& map, 
              struct img_time& img_zed, bool& run);
cv::Mat slMat2cvMat(sl::Mat& input);
#endif