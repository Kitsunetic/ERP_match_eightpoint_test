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

int test_angle_gen()
{ 
    static int i = 0;
    return i++;
}

int main()
{
    // input image
    Mat im_left = imread("left_building2.jpg");
    Mat im_right = imread("right_building2.jpg");

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
    }

    // Test with multi-angle
    vector<double> test_angle(16);
    generate(test_angle.begin(), test_angle.end(), test_angle_gen);
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

                // Image rotation test
                // We want to rotate "camera" axis, not "pixel".
                // In that case, We use inverse of rotation matrix 
                // Only rotate right image
                Mat im_right2 = erp_rot.rotate_image(im_right, rot_mat_inv);

                imshow_resize("left", im_left);
                imshow_resize("right with rotation matrix", im_right2);

                // Spherical surf test
                cout << "Spherical surf test" << endl;
                spherical_surf sph_surf;
                vector<KeyPoint> left_key;
                vector<KeyPoint> right_key;
                int match_size;
                Mat match_output;
                int total_key_num;
                sph_surf.do_all(im_left, im_right2, left_key, right_key, match_size, match_output, total_key_num);

                cout << "match result" << endl;
                cout << "total number of keypoint: " << total_key_num << endl;
                cout << "matched: " << match_size << endl;

                cout << "Eight-Point estimation test" << endl;
                eight_point estimater;
                Vec3f rot_vec_estimated, t_vec_estimated;
                Mat rot_mat_estimated;
                estimater.find(im_left.cols, im_left.rows, left_key, right_key, rot_vec_estimated, t_vec_estimated, match_size);

                Vec3d rot_vec_estimated_double = Vec3d(rot_vec_estimated[0], rot_vec_estimated[1], rot_vec_estimated[2]);
                rot_mat_estimated = erp_rot.eular2rot(rot_vec_estimated_double);
                
                Mat result_rot_mat = rot_mat_estimated*(initial_rot_mat.inv());
                Vec3d result_rot_vec = erp_rot.rot2eular(result_rot_mat);

                cout << "R_vector: " << DEGREE(result_rot_vec) << endl;
                cout << "T_vector: " << t_vec_estimated << endl;

                log << "target_R_vector: " << test_angle_vec << endl;
                log << "eightpoint_estimated_R_vector: " << DEGREE(result_rot_vec) << endl;
                log << "eightpoint_estimated_T_vector: " << t_vec_estimated << endl;
                log << "match_size: " << match_size << endl;
            }
        }
    }

    log.close();

    return 0;
}