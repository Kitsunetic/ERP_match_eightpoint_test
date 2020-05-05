#include "epipolar_tool.hpp"
#include <numeric>

using namespace std;
using namespace cv;

epipolar_tool::epipolar_tool(std::vector<cv::KeyPoint>& left_key, std::vector<cv::KeyPoint>& right_key
                            , int im_width, int im_height, int output_width, int output_height, int test_key_num)
{
    match_size = left_key.size();

    n_key = test_key_num;
    vector<int> rand_idx_arr(match_size);
    iota(rand_idx_arr.begin(), rand_idx_arr.end(), 0);
    random_shuffle(rand_idx_arr.begin(), rand_idx_arr.end());
    random_idx.assign(rand_idx_arr.begin(), rand_idx_arr.begin()+n_key);

    color_set.push_back(Scalar(0, 0, 255));
    color_set.push_back(Scalar(0, 127, 255));
    color_set.push_back(Scalar(0, 255, 255));
    color_set.push_back(Scalar(0, 255, 0));
    color_set.push_back(Scalar(255, 0, 0));
    color_set.push_back(Scalar(135, 0, 75));
    color_set.push_back(Scalar(211, 0, 148));

    epipole_mat_width = output_width;
    epipole_mat_height = output_height;
    resize_ratio_w = double(epipole_mat_width) / double(im_width);
    resize_ratio_h = double(epipole_mat_height) / double(im_height);

    // convert pixel to radian coordinate, in unit sphere
    // x : longitude
    // y : latitude
    // and convert radian to rectangular coordinate
    for(int key_idx = 0; key_idx < n_key; key_idx++)
    {
        Point2d radian;
        radian.x = 2*M_PI*(left_key[random_idx[key_idx]].pt.x / im_width);
        radian.y = M_PI*(left_key[random_idx[key_idx]].pt.y / im_height);
        key_point_left_radian.push_back(radian);

        // For MPEG's OMAF axis
        Point3d rect;
        rect.x = -sin(key_point_left_radian[key_idx].y)*cos(key_point_left_radian[key_idx].x);
        rect.y = sin(key_point_left_radian[key_idx].y)*sin(key_point_left_radian[key_idx].x);
        rect.z = cos(key_point_left_radian[key_idx].y);
        key_point_left_rect.push_back(rect);

        right_key_.push_back(right_key[random_idx[key_idx]]);
    }

    pixel_radian.resize(epipole_mat_height);
    pixel_rect.resize(epipole_mat_height);
    for(int i = 0; i < epipole_mat_height; i++)
    {
        pixel_radian[i].resize(epipole_mat_width);
        pixel_rect[i].resize(epipole_mat_width);
    }

#ifdef _WIN32
    #pragma omp parallel for
#else
    #pragma omp parallel for collapse(2)
#endif
    for(int i = 0; i < epipole_mat_height; i++)
    {
        for(int j = 0; j < epipole_mat_width; j++)
        {
            Point2d pixel_rad_tmp;
            Point3d pixel_3d_tmp;

            pixel_rad_tmp.x = 2*M_PI*(double(j) / epipole_mat_width);
            pixel_rad_tmp.y = M_PI*(double(i) / epipole_mat_height);
            pixel_3d_tmp.x = -sin(pixel_rad_tmp.y)*cos(pixel_rad_tmp.x);
            pixel_3d_tmp.y = sin(pixel_rad_tmp.y)*sin(pixel_rad_tmp.x);
            pixel_3d_tmp.z = cos(pixel_rad_tmp.y);

            pixel_radian[i][j] = pixel_rad_tmp;
            pixel_rect[i][j] = pixel_3d_tmp;
        }
    }
}

cv::Mat epipolar_tool::draw_epipole(cv::Mat& test_E_mat)
{
    Mat epipole_mat = Mat::zeros(epipole_mat_height, epipole_mat_width, CV_8UC3);
    
#ifdef _WIN32
    #pragma omp parallel for
#else
    #pragma omp parallel for collapse(3)
#endif
    for(int i = 0; i < epipole_mat_height; i++)
    {
        for(int j = 0; j < epipole_mat_width; j++)
        {
            for(int key_idx = 0; key_idx < n_key; key_idx++)
            {
                // Draw epipole
                double p[3] = {pixel_rect[i][j].x, pixel_rect[i][j].y, pixel_rect[i][j].z};
                double l[3] = {key_point_left_rect[key_idx].x , key_point_left_rect[key_idx].y , key_point_left_rect[key_idx].z};
                double* e = (double*)test_E_mat.data;
                double result = l[0]*(p[0]*e[0] + p[1]*e[3] + p[2]*e[6])
                               +l[1]*(p[0]*e[1] + p[1]*e[4] + p[2]*e[7])
                               +l[2]*(p[0]*e[2] + p[1]*e[5] + p[2]*e[8]);

                if(abs(result) < 0.002)
                {
                    epipole_mat.at<Vec3b>(i, j) = Vec3b(color_set[key_idx].val[0], color_set[key_idx].val[1], color_set[key_idx].val[2]);
                }

                // Draw feature point of right iamge
                int i_idx = right_key_[key_idx].pt.y*resize_ratio_h;
                int j_idx = right_key_[key_idx].pt.x*resize_ratio_w;
                Vec3b color_vec3b = Vec3b(color_set[key_idx].val[0], color_set[key_idx].val[1], color_set[key_idx].val[2]);
                int dot_size = 5;
                for(int dot_i = (i_idx - dot_size); dot_i < (i_idx + dot_size+1); dot_i++)
                {
                    for(int dot_j = (j_idx - dot_size); dot_j < (j_idx + dot_size+1); dot_j++)
                    {
                        epipole_mat.at<Vec3b>(dot_i, dot_j) = color_vec3b;
                    }
                }
            }
        }
    }
    return epipole_mat;
}
