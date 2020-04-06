#pragma once

#include "debug_print.h"
#include "erp_rotation.hpp"
#include <opencv2/opencv.hpp>
#include <numeric> 

class eight_point
{
    public:
    void find(int im_width, int im_height 
                       , std::vector<cv::KeyPoint>& key_left, std::vector<cv::KeyPoint>& key_right
                       , cv::Vec3f& R_vec_out, cv::Vec3f& T_vec_out
                       , int match_size);
    void eight_point_estimation(int im_width, int im_height
                            , std::vector<cv::Point3d>& key_point_left_rect, std::vector<cv::Point3d>& key_point_right_rect
                            , cv::Vec3f& R1_vec, cv::Vec3f& R2_vec, cv::Vec3f& T_vec
                            , bool& R1_valid, bool& R2_valid
                            , int match_size);
    void initial_guess(int im_width, int im_height
                    , std::vector<cv::Point3d>& key_point_left_rect, std::vector<cv::Point3d>& key_point_right_rect
                    , cv::Vec3f& R_vec_out, cv::Vec3f& T_vec_out
                    , int match_size);
    private:
    erp_rotation erp_rot;

    double max_vec(cv::Vec3f& vec);
};

class random_array
{
    public:
    random_array(int size)
    : rand_arr(size)
    {
        size_ = size;
        count_ = 0;
        rand_idx_generate();
    }

    int get_rand()
    {
        int retval = rand_arr[count_];
        count_++;
        count_ = count_ % size_;
        return retval;
    }

    private:
    int size_;
    std::vector<int> rand_arr;
    int count_;

    void rand_idx_generate()
    {
        iota(rand_arr.begin(), rand_arr.end(), 0);
        random_shuffle(rand_arr.begin(), rand_arr.end());
    }
};