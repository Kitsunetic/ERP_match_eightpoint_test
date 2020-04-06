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

#define TEST_TYPE 2
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

        log << "initial_R_vector: " << DEGREE(initial_rot_vec) << endl;
        log << "initial_T_vector: " << initial_t_vec << endl;

        Mat im_left_rectification;
        Mat im_right_rectification;

        rectify(im_left, im_right, initial_rot_vec, initial_t_vec,
                im_left_rectification, im_right_rectification);

        imshow_resize("left_rectified", im_left_rectification);
        imshow_resize("right_rectified", im_right_rectification);

        imwrite("left_origin.png", im_left);
        imwrite("right_origin.png", im_right);
        imwrite("left_rectified.png", im_left_rectification);
        imwrite("right_rectified.png", im_right_rectification);
    }

#if TEST_TYPE == 0
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
#elif TEST_TYPE == 1
    // Test with multi-angle
    vector<Vec3d> test_angles = {//Vec3d(15, 0, 0)
                                //, Vec3d(0, 15, 0)
                                //, Vec3d(0, 0, 15)
                                //, Vec3d(15, 15, 0)
                                //, Vec3d(15, 0, 15)
                                //, Vec3d(0, 15, 15)
                                 Vec3d(15, 15, 15)};
    vector<int> feature_num_limits = {100, 50, 40, 30, 20};

    for(int feature_num_limit_idx = 0; feature_num_limit_idx < feature_num_limits.size(); feature_num_limit_idx++)
    {
        log << "\nfeature_num_limit: " << feature_num_limits[feature_num_limit_idx] << endl;
        for(int test_angle_idx = 0; test_angle_idx < test_angles.size(); test_angle_idx++)
        {
            Vec3d test_angle_vec = test_angles[test_angle_idx];

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
            cout << "match limit: " << feature_num_limits[feature_num_limit_idx] << endl;

            int match_size_limited = (feature_num_limits[feature_num_limit_idx] < match_size)
                                    ? feature_num_limits[feature_num_limit_idx]
                                    : match_size;

            cout << "Eight-Point estimation test" << endl;
            eight_point estimater;
            Vec3f rot_vec_estimated, t_vec_estimated;
            Mat rot_mat_estimated;
            estimater.find(im_left.cols, im_left.rows, left_key, right_key, rot_vec_estimated, t_vec_estimated, match_size_limited);

            Vec3d rot_vec_estimated_double = Vec3d(rot_vec_estimated[0], rot_vec_estimated[1], rot_vec_estimated[2]);
            rot_mat_estimated = erp_rot.eular2rot(rot_vec_estimated_double);
            
            Mat result_rot_mat = rot_mat_estimated*(initial_rot_mat.inv());
            Vec3d result_rot_vec = erp_rot.rot2eular(result_rot_mat);

            Mat im_left_rectification;
            Mat im_right_rectification;
            rectify(im_left, im_right2, result_rot_vec, t_vec_estimated,
            im_left_rectification, im_right_rectification);

            imshow_resize("left_rectified", im_left_rectification);
            imshow_resize("right_rectified", im_right_rectification);

            cout << "R_vector: " << DEGREE(result_rot_vec) << endl;
            cout << "T_vector: " << t_vec_estimated << endl;

            log << "target_R_vector: " << test_angle_vec << endl;
            log << "eightpoint_estimated_R_vector: " << DEGREE(result_rot_vec) << endl;
            log << "eightpoint_estimated_T_vector: " << t_vec_estimated << endl;
            log << "match_size: " << match_size_limited << endl;
        }
    }
#endif

    log.close();

    return 0;
}