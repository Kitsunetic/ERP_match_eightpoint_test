#include "erp_rotation.hpp"
#include "spherical_surf.hpp"
#include "eight_point.hpp"
#include "epipolar_tool.hpp"
#include "debug_print.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <fstream>
#include <ctime>

#define OUTPUT_WIDTH 1920
#define OUTPUT_HEIGHT 1080

using namespace std;
using namespace cv;

void imshow_out(String name, Mat& out)
{
    Mat tmp;
    double h_ratio = OUTPUT_HEIGHT/out.rows;
    resize(out, tmp, Size(), h_ratio, h_ratio);
    imshow(name, tmp);
}

int main()
{
    // input image
    Mat im = imread("test_image.JPG");

    // image rotation tool
    erp_rotation erp_rot;

    // Test with multi-angle
    Vec3d test_angle_vec = Vec3d(30, 0, 0);

    // input rotation in xyz-eular
    Vec3d   rot_vec = RAD(test_angle_vec);
    Mat     rot_mat = erp_rot.eular2rot(rot_vec);
    Mat     rot_mat_inv = rot_mat.inv();
    
    // Image rotation test
    // We want to rotate "camera" axis, not "pixel".
    // In that case, We use inverse of rotation matrix 
    Mat im2 = erp_rot.rotate_image(im, rot_mat_inv);

    // Spherical surf test
    cout << "Spherical surf test" << endl;
    spherical_surf sph_surf;
    vector<KeyPoint> left_key;
    vector<KeyPoint> right_key;
    int match_size;
    Mat match_output;
    int total_key_num;
    sph_surf.do_all(im, im2, left_key, right_key, match_size, match_output, total_key_num);

    cout << "match result" << endl;
    cout << "total number of keypoint: " << total_key_num << endl;
    cout << "matched: " << match_size << endl;

    // Tool for drawing epipolar line
    cout << "Draw epipole line on ERP" << endl;
    epipolar_tool epi_tool(left_key, right_key, im.cols, im.rows, OUTPUT_WIDTH/10, OUTPUT_HEIGHT/10, 7);

    int x_angle = 30, y_angle = 30, z_angle = 30;
    String test_window_name ="Epipole manual tester";
    namedWindow(test_window_name);
    createTrackbar("x_axis_rotation", test_window_name, &x_angle, 60);
    createTrackbar("y_axis_rotation", test_window_name, &y_angle, 60);
    createTrackbar("z_axis_rotation", test_window_name, &z_angle, 60);
    setTrackbarPos("x_axis_rotation", test_window_name, 30);
    setTrackbarPos("y_axis_rotation", test_window_name, 30);
    setTrackbarPos("z_axis_rotation", test_window_name, 30);

    // Draw epipolar line
    while(1)
    {
        // Test Essential matrix
        Vec3d test_rot_vec = RAD(Vec3d(x_angle-30, y_angle-30, z_angle-30));
        Vec3d test_t_vec = Vec3d(0, -1, 0);

        Mat test_rot_mat = erp_rot.eular2rot(test_rot_vec);
        Mat test_t_cross = (Mat_<double>(3, 3) << 0 , -test_t_vec[2] , test_t_vec[1]
                                            , test_t_vec[2] , 0 , -test_t_vec[0]
                                            , -test_t_vec[1] , test_t_vec[0] , 0);
        Mat test_E_mat = test_rot_mat.inv()*test_t_cross;

        Mat epipole_mat = epi_tool.draw_epipole(test_E_mat);
        imshow(test_window_name, epipole_mat);

        if(waitKey(1) == 27)
            break;
    }

    return 0;
}