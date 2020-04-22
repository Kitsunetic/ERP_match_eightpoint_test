#pragma once

#include "erp_rotation.hpp"
#include "eight_point.hpp"
#include "INIReader.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <fstream>
#include <ctime>
#include <sstream>

typedef struct 
{
    std::string im_left_name;
    std::string im_right_name;
    int resize_input;
    int resize_input_width;
    int resize_input_height;
    int output_height;
    int mouse_offset_max;
    int mouse_window_max;
    int mouse_window_min;
    std::string window_name;
    std::string mouse_window_name;
    std::string debug_window_name;
    int debug_window_width;
    int debug_window_height;
} enviroment_setup_t;

typedef struct
{
    cv::Point2i pt;
    double resize_ratio;
    int x_offset;
    int y_offset;
    int magnifying_size;
    std::vector<cv::Point2d> click_points_left;
    std::vector<cv::Point2d> click_points_right;
    int point_state;
} callback_param_t;

enum
{
    LEFT_POINT = 0,
    RIGHT_POINT,
};