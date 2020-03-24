#include "erp_rotation.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>

using namespace std;
using namespace cv;

#define DEBUG_IMSHOW

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

int main()
{
    // input image
    Mat im = imread("test_image.JPG");

    // image rotation tool
    erp_rotation erp_rot;

    // Test with multi-angle
    Vec3d rot_vec_x_axis = RAD(Vec3d(30, 0, 0));
    Vec3d rot_vec_y_axis = RAD(Vec3d(0, 30, 0));
    Vec3d rot_vec_z_axis = RAD(Vec3d(0, 0, 30));
    Vec3d rot_vec_yz_axis = RAD(Vec3d(0, 30, 30));
    Vec3d rot_vec_xyz_axis = RAD(Vec3d(30, 30, 30));

    vector<Vec3d> rot_vec = {rot_vec_x_axis
                            , rot_vec_y_axis
                            , rot_vec_z_axis
                            , rot_vec_yz_axis
                            , rot_vec_xyz_axis};
    vector<string> rot_name = {"rot_vec_x_axis"
                            , "rot_vec_y_axis"
                            , "rot_vec_z_axis"
                            , "rot_vec_yz_axis"
                            , "rot_vec_xyz_axis"};

    imshow_resize("original", im);
    for(int i = 0; i < rot_vec.size(); i++)
    {
        // input rotation in xyz-eular
        Mat     rot_mat = erp_rot.eular2rot(rot_vec[i]);
        Mat     rot_mat_inv = rot_mat.inv();

        // Image rotation test
        // We want to rotate "camera" axis, not "pixel".
        // In that case, We use inverse of rotation matrix 
        Mat im2 = erp_rot.rotate_image(im, rot_mat_inv);

        imshow_resize(rot_name[i], im2);
    }

    return 0;
}