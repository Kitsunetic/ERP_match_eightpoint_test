#pragma once

#include "debug_print.h"
#include <opencv2/opencv.hpp>

#define RAD(x) M_PI*(x)/180.0
#define DEGREE(x) 180.0*(x)/M_PI

class erp_rotation
{
    public:

    cv::Mat eular2rot(cv::Vec3d theta);
    cv::Vec3d rot2eular(cv::Mat R);
    cv::Vec2i rotate_pixel(const cv::Vec2i& in_vec, cv::Mat& rot_mat, int width, int height);
    cv::Mat rotate_image(const cv::Mat& im, cv::Mat& rot_mat);

    private:
};
