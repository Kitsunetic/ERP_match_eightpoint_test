#include "eight_point.hpp"

using namespace std;
using namespace cv;

double eight_point::max_vec(Vec3f& vec)
{
    if((vec[0] > vec[1]) && (vec[0] > vec[2]))
        return vec[0];
    else if(vec[1] > vec[2])
        return vec[1];
    else
        return vec[2];  
}

void eight_point::eight_point_estimation(int im_width, int im_height
                            , vector<Point3d>& key_point_left_rect, vector<Point3d>& key_point_right_rect
                            , Vec3f& R1_vec, Vec3f& R2_vec, Vec3f& T_vec
                            , bool& R1_valid, bool& R2_valid
                            , int match_size)
{
    Mat A_mat(match_size, 9, CV_64FC1);
    Mat w, u, vt;
    double* A_data = (double*)A_mat.data;
    #pragma omp parallel for
    for(int i = 0; i < match_size; i++)
    {
        A_data[i * A_mat.cols + 0] = key_point_left_rect[i].x * key_point_right_rect[i].x;
        A_data[i * A_mat.cols + 1] = key_point_left_rect[i].x * key_point_right_rect[i].y;
        A_data[i * A_mat.cols + 2] = key_point_left_rect[i].x * key_point_right_rect[i].z;
        A_data[i * A_mat.cols + 3] = key_point_left_rect[i].y * key_point_right_rect[i].x;
        A_data[i * A_mat.cols + 4] = key_point_left_rect[i].y * key_point_right_rect[i].y;
        A_data[i * A_mat.cols + 5] = key_point_left_rect[i].y * key_point_right_rect[i].z;
        A_data[i * A_mat.cols + 6] = key_point_left_rect[i].z * key_point_right_rect[i].x;
        A_data[i * A_mat.cols + 7] = key_point_left_rect[i].z * key_point_right_rect[i].y;
        A_data[i * A_mat.cols + 8] = key_point_left_rect[i].z * key_point_right_rect[i].z;
    }

    SVDecomp(A_mat, w, u, vt);

    // Find E matrix and correct Error
    Mat e_mat = vt.row(vt.rows-1);
    double* e_data = (double*)e_mat.data;
    Mat E_mat = (Mat_<double>(3, 3) << e_data[0], e_data[1], e_data[2], e_data[3], e_data[4], e_data[5], e_data[6], e_data[7], e_data[8]);
    Mat w_f, u_f, vt_f;
    SVDecomp(E_mat, w_f, u_f, vt_f);
    w_f.at<double>(0, 2) = 0.0;
    double* w_f_data = (double*)w_f.data;
    Mat w_f_diag = (Mat_<double>(3, 3) << w_f_data[0], 0, 0, 0, w_f_data[1], 0, 0, 0, w_f_data[2]);
    Mat E_mat_correct = u_f * w_f_diag * vt_f;

    // get R|t from E matrix, t is unit vector
    Mat R1, R2, t;
    decomposeEssentialMat(E_mat_correct, R1, R2, t);

    // Debug and logging
    R1_vec = erp_rot.rot2eular(R1);
    R2_vec = erp_rot.rot2eular(R2);
    T_vec[0] = t.at<double>(0,0);
    T_vec[1] = t.at<double>(1,0);
    T_vec[2] = t.at<double>(2,0);

    int test_num = 5;
    vector<Point3d> key_point_left_test(test_num);
    vector<Point3d> key_point_right_test_1(test_num);
    vector<Point3d> key_point_right_test_2(test_num);
    Mat R1_inv = R1.inv();
    Mat R2_inv = R2.inv();
    
    // Assume each rotations are under 90degree, almost 1.57 rad
    // determin valid or not
    Vec3f R1_vec_abs(abs(R1_vec[0]), abs(R1_vec[1]), abs(R1_vec[2]));
    Vec3f R2_vec_abs(abs(R2_vec[0]), abs(R2_vec[1]), abs(R2_vec[2]));
    double R1_vec_max = max_vec(R1_vec_abs);
    double R2_vec_max = max_vec(R2_vec_abs);
    if(R1_vec_max < 1.57)
        R1_valid = true;
    else
        R1_valid = false;
    
    if(R2_vec_max < 1.57)
        R2_valid = true;
    else
        R2_valid = false;
}

void eight_point::initial_guess(int im_width, int im_height
                    , vector<Point3d>& key_point_left_rect, vector<Point3d>& key_point_right_rect
                    , Vec3f& R_vec_out, Vec3f& T_vec_out
                    , int match_size)
{
    // Estimate R and T with 8 point algorithm
    Vec3f R1_vec, R2_vec, T_vec;
    bool R1_valid, R2_valid;
    vector<Vec3f> R_vec_arr;
    vector<Vec3f> T_vec_arr;
    DEBUG_PRINT_OUT("E matrix estimation with SVD");
    // Choose random M point, do 8 point algorithm, check result and find R|t
    for(int i = 0; i < 80; i++)
    {
        random_array rand_arr(match_size);
        int sample_n = match_size*0.25;
        vector<Point3d> key_point_left_rand(sample_n);
        vector<Point3d> key_point_right_rand(sample_n);
        for(int i = 0; i < sample_n; i++)
        {
            unsigned int idx;
            idx = rand_arr.get_rand();
            key_point_left_rand[i] = key_point_left_rect[idx];
            key_point_right_rand[i] = key_point_right_rect[idx];
        }
        eight_point_estimation(im_width, im_height
                               , key_point_left_rand, key_point_right_rand
                               , R1_vec, R2_vec, T_vec
                               , R1_valid, R2_valid
                               , sample_n);
        if(R1_valid)
        {
            R_vec_arr.push_back(R1_vec);
            T_vec_arr.push_back(T_vec);
        }
        if(R2_valid)
        {
            R_vec_arr.push_back(R2_vec);
            T_vec_arr.push_back(T_vec);
        }
    }

    // check distance with another R_vectors, simply check euclidian distance
    // Find minimum distnace - R_vector. must be well estimated one
    int r_vec_size = R_vec_arr.size();
    vector<double> dist(r_vec_size);
    for(int i = 0; i < r_vec_size; i++)
    {
        vector<double> diffnorm_arr(r_vec_size);
        for(int j = 0; j < r_vec_size; j++)
        {
            Vec3f vec_diff = R_vec_arr[i] - R_vec_arr[j];
            double diffnorm = sqrt(vec_diff[0]*vec_diff[0] + vec_diff[1]*vec_diff[1] + vec_diff[2]*vec_diff[2]);
            diffnorm_arr[j] = diffnorm;
        }
        sort(diffnorm_arr.begin(), diffnorm_arr.end());
        vector<double> diffnorm_arr_sub(diffnorm_arr.begin() + r_vec_size*0.2, diffnorm_arr.begin() + r_vec_size*0.8);
        double diff_avg = std::accumulate(diffnorm_arr_sub.begin(), diffnorm_arr_sub.end(), 0.0) / (diffnorm_arr_sub.size()*1.0);
        dist[i] = diff_avg;
    }
    int min_idx = std::distance(dist.begin(), std::min_element(dist.begin(), dist.end()));
    R_vec_out = R_vec_arr[min_idx];
    T_vec_out = T_vec_arr[min_idx];
}

void eight_point::find(int im_width, int im_height 
                       , vector<KeyPoint>& key_left, vector<KeyPoint>& key_right
                       , Vec3f& R_vec_out, Vec3f& T_vec_out
                       , int match_size)
{
    vector<Point2d> key_point_left_radian(match_size);
    vector<Point2d> key_point_right_radian(match_size);

    // convert pixel to radian coordinate, in unit sphere
    // x : longitude
    // y : latitude
    #pragma omp parallel for
    for(int i = 0; i < match_size; i++)
    {
        key_point_left_radian[i].x = 2*M_PI*(key_left[i].pt.x / im_width);
        key_point_right_radian[i].x = 2*M_PI*(key_right[i].pt.x / im_width);
        key_point_left_radian[i].y = M_PI*(key_left[i].pt.y / im_height);
        key_point_right_radian[i].y = M_PI*(key_right[i].pt.y / im_height);
    }

    // convert radian to rectangular coordinate
    vector<Point3d> key_point_left_rect(match_size);
    vector<Point3d> key_point_right_rect(match_size);
    #pragma omp parallel for
    for(int i = 0; i < match_size; i++)
    {
        key_point_left_rect[i].x = sin(key_point_left_radian[i].y)*cos(key_point_left_radian[i].x);
        key_point_left_rect[i].y = sin(key_point_left_radian[i].y)*sin(key_point_left_radian[i].x);
        key_point_left_rect[i].z = cos(key_point_left_radian[i].y);

        key_point_right_rect[i].x = sin(key_point_right_radian[i].y)*cos(key_point_right_radian[i].x);
        key_point_right_rect[i].y = sin(key_point_right_radian[i].y)*sin(key_point_right_radian[i].x);
        key_point_right_rect[i].z = cos(key_point_right_radian[i].y);
    }

    initial_guess(im_width, im_height
                    , key_point_left_rect, key_point_right_rect
                    , R_vec_out, T_vec_out
                    , match_size);
}
