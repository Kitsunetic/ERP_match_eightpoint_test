#include "erp_rotation.hpp"
#include "spherical_surf.hpp"
#include "eight_point.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <fstream>
#include <ctime>
#include <omp.h>

#include "debug_print.h"

using namespace std;
using namespace cv;


static double degree_error(Point2f vec1, Point2f vec2, int width, int height)
{
    //(x,y) pixel coordinate to (row,col) radian
    Vec2d vec1_rad = Vec2d(M_PI*vec1.y/height, 2*M_PI*vec1.x/width);
    Vec2d vec2_rad = Vec2d(M_PI*vec2.y/height, 2*M_PI*vec2.x/width);

    //(x,y,z) cartesian coordinate, quantity 1 vector 
    Vec3d vec1_cartesian;
    vec1_cartesian[0] = sin(vec1_rad[0])*cos(vec1_rad[1]);
    vec1_cartesian[1] = sin(vec1_rad[0])*sin(vec1_rad[1]);
    vec1_cartesian[2] = cos(vec1_rad[0]);
    Vec3d vec2_cartesian;
    vec2_cartesian[0] = sin(vec2_rad[0])*cos(vec2_rad[1]);
    vec2_cartesian[1] = sin(vec2_rad[0])*sin(vec2_rad[1]);
    vec2_cartesian[2] = cos(vec2_rad[0]);

    double product = vec1_cartesian[0]*vec2_cartesian[0]
                    +vec1_cartesian[1]*vec2_cartesian[1]
                    +vec1_cartesian[2]*vec2_cartesian[2];
    double error_rad = 0;
    if(product < 1)
        error_rad = acos(product);
    return error_rad;
}

int test_angle_gen()
{ 
    static int i = 0;
    return i++;
}

Mat rot_from_vec(Vec3d vec1, Vec3d vec2)
{
    Vec3d v = Vec3d(vec1[1]*vec2[2] - vec1[2]*vec2[1]
                   , vec1[2]*vec2[0] - vec1[0]*vec2[2]
                   , vec1[0]*vec2[1] - vec1[1]*vec2[0]);
    double c = vec1[0]*vec2[0] + vec1[1]*vec2[1] + vec1[2]*vec2[2];

    Mat v_cross = (Mat_<double>(3, 3) << 0, -v[2], v[1]
                                      ,v[2], 0, -v[0]
                                      ,-v[1], v[0], 0);
    Mat I_mat = Mat::eye(3, 3, CV_64F);
    Mat R = I_mat + v_cross + v_cross*v_cross*(1/1+c);

    return R;
}

void rectify(const Mat& im_left, const Mat& im_right, Vec3d initial_rot_vec, Vec3d initial_t_vec,
                 Mat& im_left_rectification, Mat& im_right_rectification)
{    
    // image rotation tool
    erp_rotation erp_rot;

    Mat R_left_recti = rot_from_vec(Vec3d(0, -1, 0), initial_t_vec);
    Mat R_left_recti_inv = R_left_recti.inv();
    Mat R_right_recti = R_left_recti*erp_rot.eular2rot(initial_rot_vec).inv();
    Mat R_right_recti_inv = R_right_recti.inv();

    im_left_rectification = erp_rot.rotate_image(im_left, R_left_recti_inv);
    im_right_rectification = erp_rot.rotate_image(im_right, R_right_recti_inv);
}

int main(int argc, char* argv[])
{
    if(argc != 3)
    {
        cout << "usage: " << argv[0] << " <left image> <right image>" << endl;
        return 0;
    }
    // input image
    string im_left_name = argv[1];
    string im_right_name = argv[2];
    Mat im_left = imread(im_left_name);
    Mat im_right = imread(im_right_name);

    // image rotation tool
    erp_rotation erp_rot;

    // log file open
    string log_name = "estimated_extrinsic.txt";
    ofstream log;
    log.open(log_name);

    Vec3d initial_rot_vec;
    Vec3d initial_t_vec;
    Mat initial_rot_mat;
    {
        // Initial Guess
        cout << "initial Guess" << endl;

        // Spherical surf test
        cout << "Spherical surf test" << endl;
        spherical_surf sph_surf;
        vector<KeyPoint> left_key;
        vector<KeyPoint> right_key;
        int match_size;
        Mat match_output;
        int total_key_num;
        sph_surf.do_all(im_left, im_right, left_key, right_key, match_size, match_output, total_key_num);

        cout << "match result" << endl;
        cout << "total number of keypoint: " << total_key_num << endl;
        cout << "matched: " << match_size << endl;

        cout << "Eight-Point estimation test" << endl;
        eight_point estimater;
        Vec3f rot_vec_estimated, t_vec_estimated;
        estimater.find(im_left.cols, im_left.rows, left_key, right_key, rot_vec_estimated, t_vec_estimated, match_size);

        initial_rot_vec = Vec3d(rot_vec_estimated[0], rot_vec_estimated[1], rot_vec_estimated[2]);
        initial_t_vec = Vec3d(t_vec_estimated[0], t_vec_estimated[1], t_vec_estimated[2]);
        initial_rot_mat = erp_rot.eular2rot(initial_rot_vec);

        cout << "R_vector: " << DEGREE(initial_rot_vec) << endl;
        cout << "T_vector: " << initial_t_vec << endl;

        log << "initial_R_vector: " << DEGREE(initial_rot_vec) << endl;
        log << "initial_T_vector: " << initial_t_vec << endl;

        Mat im_left_rectification;
        Mat im_right_rectification;
        
        START_TIME(__rectify__);
        rectify(im_left, im_right, initial_rot_vec, initial_t_vec,
                im_left_rectification, im_right_rectification);
        STOP_TIME(__rectify__);
        
        START_TIME(__ERP_ROT__);
        erp_rotation erp_rot;
        Mat rot_mat_90deg = erp_rot.eular2rot(Vec3d(RAD(89.999), 0, 0)).inv();
        Mat left_rotate = erp_rot.rotate_image(im_left_rectification, rot_mat_90deg);
        Mat right_rotate = erp_rot.rotate_image(im_right_rectification, rot_mat_90deg);
        cv::rotate(left_rotate, left_rotate, cv::ROTATE_90_CLOCKWISE);
        cv::rotate(right_rotate, right_rotate, cv::ROTATE_90_CLOCKWISE);
        STOP_TIME(__ERP_ROT__);
        
        START_TIME(__IMWRITE__);
        imwrite("rectified_left.png", im_left_rectification);
        imwrite("rectified_right.png", im_right_rectification);
        imwrite("rectified_left_vertical.png", left_rotate);
        imwrite("rectified_right_vertical.png", right_rotate);
        STOP_TIME(__IMWRITE__);
    }

    log.close();

    return 0;
}
