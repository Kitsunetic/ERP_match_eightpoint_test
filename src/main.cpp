#include "erp_rotation.hpp"
#include "spherical_surf.hpp"
#include "eight_point.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

//#define DEBUG_IMSHOW

#ifdef DEBUG_IMSHOW
static void imshow_resize(String name, Mat& im)
{
    Mat tmp;
    resize(im, tmp, Size(), 0.25, 0.25);
    imshow(name, tmp);
    waitKey(0);
}
#else
static void imshow_resize(String name, Mat& im){}
#endif
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

int main()
{
    // input image
    Mat im = imread("test_image.JPG");

    // image rotation tool
    erp_rotation erp_rot;

    // log file open
    time_t now = time(0);
    tm *local_time = localtime(&now);
    string log_name = to_string(local_time->tm_year)
                      +'_'+to_string(local_time->tm_mon)
                      +'_'+to_string(local_time->tm_mday)
                      +'_'+to_string(local_time->tm_hour)
                      +'_'+to_string(local_time->tm_min)
                      +"_log.txt";
    ofstream log;
    log.open(log_name);

    // Test with multi-angle
    vector<double> test_angle = {0, 5, 10, 15, 20}; 
    for(int x = 0; x < test_angle.size(); x++)
    {
        for(int y = 0; y < test_angle.size(); y++)
        {
            for(int z = 0; z < test_angle.size(); z++)
            {
                Vec3d test_angle_vec = Vec3d(test_angle[x], test_angle[y], test_angle[z]);

                // input rotation in xyz-eular
                Vec3d   rot_vec = RAD(test_angle_vec);
                Mat     rot_mat = erp_rot.eular2rot(rot_vec);
                Mat     rot_mat_inv = rot_mat.inv();

                // Rotation matrix <-> vector conversion test in xyz-eular
                cout << "Rotation matrix <-> vector conversion test in xyz-eular" << endl;
                Vec3d   rot_vec_2 = erp_rot.rot2eular(rot_mat);
                cout << "Test rotation vector input, euler-XYZ, " << rot_vec << endl;
                cout << "Test rotation vector output, euler-XYZ, " << rot_vec_2 << endl;

                // Image rotation test
                Mat im2 = erp_rot.rotate_image(im, rot_mat);

                imshow_resize("original", im);
                imshow_resize("with rotation matrix", im2);

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

                // match error test
                Mat keypoint_valid_mat = im2.clone();
                vector<double> error(match_size);
                for(int i = 0; i < match_size; i++)
                {
                    Vec2i left_key_vec = Vec2i(left_key[i].pt.y, left_key[i].pt.x);
                    Vec2i left_key_rotate = erp_rot.rotate_pixel(left_key_vec, rot_mat, im.cols, im.rows);
                    Point2f left_key_rotate_pt = Point2f(left_key_rotate[1], left_key_rotate[0]);

                    circle(keypoint_valid_mat, right_key[i].pt, 4, Scalar(0, 255, 0),2);
                    circle(keypoint_valid_mat, left_key_rotate_pt, 4, Scalar(255, 0, 0),2);

                    error[i] = degree_error(left_key_rotate_pt, right_key[i].pt, im.cols, im.rows);
                }
                double average = std::accumulate( error.begin(), error.end(), 0.0) / match_size;

                cout << "Surf match error, degree, mean : " << average << endl;
                imshow_resize("keypoint_valid_mat", keypoint_valid_mat);

                cout << "Eight-Point estimation test" << endl;
                eight_point estimater;
                Vec3f rot_vec_estimated, t_vec_estimated;
                estimater.find(im.cols, im.rows, left_key, right_key, rot_vec_estimated, t_vec_estimated, match_size);
                
                cout << "R_vector: " << DEGREE(rot_vec_estimated) << endl;
                cout << "T_vector: " << t_vec_estimated << endl;

                log << "target_R_vector: " << test_angle_vec << endl;
                log << "eightpoint_estimated_R_vector: " << DEGREE(rot_vec_estimated) << endl;
                log << "surf_match_error: " << average << endl;
            }
        }
    }

    log.close();

    return 0;
}