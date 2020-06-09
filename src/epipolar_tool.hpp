#pragma once

#include "debug_print.h"
#include <opencv2/opencv.hpp>
#include <vector>

class epipolar_tool
{
    public:
    epipolar_tool(std::vector<cv::KeyPoint>& left_key, std::vector<cv::KeyPoint>& right_key
                     , int im_width, int im_height, int output_width, int output_height, int test_key_num);
    cv::Mat draw_epipole(cv::Mat& test_E_mat);

    private:
    int match_size;
    std::vector<int> random_idx;
    std::vector<cv::Scalar> color_set;
    std::vector<cv::Point2d> key_point_left_radian;
    std::vector<cv::Point3d> key_point_left_rect;
    std::vector<std::vector<cv::Point2d>> pixel_radian;
    std::vector<std::vector<cv::Point3d>> pixel_rect;
    std::vector<cv::KeyPoint> right_key_;
    int epipole_mat_width;
    int epipole_mat_height;
    double resize_ratio_w;
    double resize_ratio_h;
    int n_key;
};
