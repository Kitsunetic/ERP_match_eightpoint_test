#include "erp_rotation.hpp"
#include "eight_point.hpp"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <numeric>
#include <fstream>
#include <ctime>

using namespace std;
using namespace cv;

#define OUTPUT_HEIGHT 960
#define CROP_WINDOW_SIZE 11
#define MOUSE_OFFSET_MAX 3
#define BORDER ((CROP_WINDOW_SIZE/2) + MOUSE_OFFSET_MAX)
#define WINDOW_NAME "test_show"
#define MOUSE_WINDOW_NAME "mouse_point_magnification"

static void imshow_resize(Mat& im_left, Mat& im_right, Point2i mouse_pt)
{
    double resize_ratio = double(OUTPUT_HEIGHT)/double(im_left.rows*2);
    Mat im_left_resize, im_right_resize;
    resize(im_left, im_left_resize, Size(), resize_ratio, resize_ratio);
    resize(im_right, im_right_resize, Size(), resize_ratio, resize_ratio);
    vector<Mat> show_vector = {im_left_resize, im_right_resize};
    Mat show_mat;
    vconcat(show_vector, show_mat);
    imshow(WINDOW_NAME, show_mat);

    Mat mouse_window;
    Mat im_left_border, im_right_border;
    copyMakeBorder(im_left, im_left_border
                   , BORDER, BORDER, BORDER, BORDER
                   , BORDER_CONSTANT);
    copyMakeBorder(im_right, im_right_border
                   , BORDER, BORDER, BORDER, BORDER
                   , BORDER_CONSTANT);
    if(mouse_pt.y < im_left.rows)
    {
        int x = BORDER + mouse_pt.x;
        int y = BORDER + mouse_pt.y;
        mouse_window = im_left_border(Rect(x-(CROP_WINDOW_SIZE/2)
                                            , y-(CROP_WINDOW_SIZE/2)
                                            , CROP_WINDOW_SIZE
                                            , CROP_WINDOW_SIZE));
    }
    else
    {
        int x = BORDER + mouse_pt.x;
        int y = BORDER + mouse_pt.y - im_left.rows;
        mouse_window = im_right_border(Rect(x-(CROP_WINDOW_SIZE/2)
                                            , y-(CROP_WINDOW_SIZE/2)
                                            , CROP_WINDOW_SIZE
                                            , CROP_WINDOW_SIZE));
    }
    mouse_window.at<Vec3b>((CROP_WINDOW_SIZE/2), (CROP_WINDOW_SIZE/2)) = Vec3b(0, 255, 0);
    resize(mouse_window, mouse_window, Size(200, 200), 0, 0, CV_INTER_NN);
    imshow(MOUSE_WINDOW_NAME, mouse_window);
}

enum
{
    LEFT_POINT = 0,
    RIGHT_POINT,
};
static int point_state = LEFT_POINT;
static vector<Point2d> click_points_left;
static vector<Point2d> click_points_right;
typedef struct
{
    Point2i pt;
    double resize_ratio;
    int x_offset;
    int y_offset;
} callback_param_t;
static void pick_point(int evt, int x, int y, int flags, void* param)
{
    callback_param_t* callback_param = (callback_param_t*)param;
    double resize_ratio = callback_param->resize_ratio;
    if(evt == CV_EVENT_MOUSEMOVE)
    {
        callback_param->pt = Point2i(x, y)/resize_ratio;
        callback_param->pt.x += callback_param->x_offset;
        callback_param->pt.y += callback_param->y_offset;
    }
    else if(evt == CV_EVENT_LBUTTONDOWN)
    {
        Rect window_rect = getWindowImageRect(WINDOW_NAME);
        if(point_state == LEFT_POINT)
        {
            if(y < window_rect.height/2)
            {
                cout << "left point input" << endl;
                Point2i input = callback_param->pt;
                click_points_left.push_back(input);
                cout << "Point x: " << input.x << " Point y: " << input.y << endl;
                cout << "num of left point: " << click_points_left.size() << endl;
                cout << "num of right point: " << click_points_right.size() << endl;
                point_state = RIGHT_POINT;
            }
            else
                cout << "Not this area" << endl;
        }
        else if(point_state == RIGHT_POINT)
        {
            if(y < window_rect.height/2)
                cout << "Not this area" << endl;
            else
            {
                cout << "right point input" << endl;
                Point2i input = Point2i(callback_param->pt.x, callback_param->pt.y - (window_rect.height/2)/resize_ratio);
                click_points_right.push_back(input);
                cout << "Point x: " << input.x << " Point y: " << input.y << endl;
                cout << "num of left point: " << click_points_left.size() << endl;
                cout << "num of right point: " << click_points_right.size() << endl;
                point_state = LEFT_POINT;
            }
        }
    }
    else if(evt == CV_EVENT_RBUTTONDOWN)
    {
        int left_size = click_points_left.size();
        int right_size = click_points_right.size();
        if((left_size == right_size)&&(left_size >= 8))
        {
            cout << "eight_point_start" << endl;
            Rect window_rect = getWindowImageRect(WINDOW_NAME);
            int match_size = left_size;
            int im_width = window_rect.width*(1/resize_ratio);
            int im_height = window_rect.height*(1/resize_ratio)/2;

            cout << "im_width: " << im_width << " im_height: " << im_height << endl;

            // convert pixel to radian coordinate, in unit sphere
            // x : longitude
            // y : latitude
            vector<Point2d> key_point_left_radian(match_size);
            vector<Point2d> key_point_right_radian(match_size);
            for(int i = 0; i < match_size; i++)
            {
                key_point_left_radian[i].x = 2*M_PI*(click_points_left[i].x / im_width);
                key_point_right_radian[i].x = 2*M_PI*(click_points_right[i].x / im_width);
                key_point_left_radian[i].y = M_PI*(click_points_left[i].y / im_height);
                key_point_right_radian[i].y = M_PI*(click_points_right[i].y / im_height);
            }

            // convert radian to rectangular coordinate
            vector<Point3d> key_point_left_rect(match_size);
            vector<Point3d> key_point_right_rect(match_size);
            for(int i = 0; i < match_size; i++)
            {
                // For MPEG's OMAF axis
                key_point_left_rect[i].x = -sin(key_point_left_radian[i].y)*cos(key_point_left_radian[i].x);
                key_point_left_rect[i].y = sin(key_point_left_radian[i].y)*sin(key_point_left_radian[i].x);
                key_point_left_rect[i].z = cos(key_point_left_radian[i].y);

                key_point_right_rect[i].x = -sin(key_point_right_radian[i].y)*cos(key_point_right_radian[i].x);
                key_point_right_rect[i].y = sin(key_point_right_radian[i].y)*sin(key_point_right_radian[i].x);
                key_point_right_rect[i].z = cos(key_point_right_radian[i].y);
            }

            eight_point ep;
            Vec3f R1_vec, R2_vec, T_vec;
            bool R1_valid, R2_valid;
            Vec3f result_R, result_T;
            ep.eight_point_estimation(im_width, im_height
                            , key_point_left_rect, key_point_right_rect
                            , R1_vec, R2_vec, T_vec
                            , R1_valid, R2_valid
                            , match_size);
            if(R1_valid)
            {
                result_R = R1_vec;
                result_T = T_vec;
            }
            if(R2_valid)
            {
                result_R = R2_vec;
                result_T = T_vec;
            }
            cout << "R1: " << R1_vec << endl;
            cout << "R2: " << R2_vec << endl;
            cout << "t: " << T_vec << endl;
            cout << "Result R vector : " << result_R << endl;
            cout << "Result_T vector : " << result_T << endl;
        }
        else
        {
            cout << "No enough point or size of left/right point not same" << endl;
        }
    }
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

    callback_param_t callback_param;
    callback_param.resize_ratio = double(OUTPUT_HEIGHT)/double(im_left.rows*2);
    double resize_ratio = double(OUTPUT_HEIGHT)/double(im_left.rows*2);
    
    namedWindow(WINDOW_NAME);
    setMouseCallback(WINDOW_NAME, pick_point, (void*)&callback_param);

    namedWindow(MOUSE_WINDOW_NAME);
    createTrackbar("x_offset", MOUSE_WINDOW_NAME, &(callback_param.x_offset), 7);
    createTrackbar("y_offset", MOUSE_WINDOW_NAME, &(callback_param.y_offset), 7);
    setTrackbarMin("x_offset", MOUSE_WINDOW_NAME, -3);
    setTrackbarMax("x_offset", MOUSE_WINDOW_NAME, 3);
    setTrackbarMin("y_offset", MOUSE_WINDOW_NAME, -3);
    setTrackbarMax("y_offset", MOUSE_WINDOW_NAME, 3);
    setTrackbarPos("x_offset", MOUSE_WINDOW_NAME, 0);
    setTrackbarPos("y_offset", MOUSE_WINDOW_NAME, 0);

    while(1)
    {
        Mat im_left_tmp = im_left.clone();
        Mat im_right_tmp = im_right.clone();
        if(click_points_left.size() > 0)
        {
            for(int i = 0; i < click_points_left.size(); i++)
                circle(im_left_tmp, click_points_left[i], 5, Scalar(255, 0, 0), 3);
        }
        if(click_points_right.size() > 0)
        {
            for(int i = 0; i < click_points_left.size(); i++)
                circle(im_right_tmp, click_points_right[i], 5, Scalar(255, 0, 0), 3);
        }
        imshow_resize(im_left_tmp, im_right_tmp, callback_param.pt);
        waitKey(10);
    }

    return 0;
}