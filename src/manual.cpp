#include "common_header.hpp"

using namespace std;
using namespace cv;

static enviroment_setup_t env_setup;

static Mat debug_mat;
void debug_show(vector<string>& str)
{
    debug_mat = Mat::zeros(Size(env_setup.debug_window_width, env_setup.debug_window_height), CV_8UC3);
    for(int i = 0; i < str.size(); i++)
    {
        putText(debug_mat, str[i], Point(0, 20*(1+i)), cv::HersheyFonts::FONT_HERSHEY_DUPLEX, 0.8, Scalar(0, 255, 0));
    }
    imshow(env_setup.debug_window_name, debug_mat);
}

static void imshow_resize(Mat& im_left, Mat& im_right)
{
    double resize_ratio = double(env_setup.output_height)/double(im_left.rows*2);
    Mat im_left_resize, im_right_resize;
    resize(im_left, im_left_resize, Size(), resize_ratio, resize_ratio);
    resize(im_right, im_right_resize, Size(), resize_ratio, resize_ratio);
    vector<Mat> show_vector = {im_left_resize, im_right_resize};
    Mat show_mat;
    vconcat(show_vector, show_mat);
    imshow(env_setup.window_name, show_mat);
}

static void mouse_show(Mat& im_left, Mat& im_right, callback_param_t& callback_param)
{
    Point2i mouse_pt = callback_param.pt;
    int magnifying_size = callback_param.magnifying_size;
    int border = magnifying_size/2;
    int window_size = border*2+1;

    Mat mouse_window;
    Mat im_left_border, im_right_border;
    copyMakeBorder(im_left, im_left_border
                   , border, border, border, border
                   , BORDER_CONSTANT);
    copyMakeBorder(im_right, im_right_border
                   , border, border, border, border
                   , BORDER_CONSTANT);
    if(mouse_pt.y < im_left.rows)
    {
        int x = border + mouse_pt.x;
        int y = border + mouse_pt.y;
        mouse_window = im_left_border(Rect(x-(window_size/2)
                                            , y-(window_size/2)
                                            , window_size
                                            , window_size));
    }
    else
    {
        int x = border + mouse_pt.x;
        int y = border + mouse_pt.y - im_left.rows;
        mouse_window = im_right_border(Rect(x-(window_size/2)
                                            , y-(window_size/2)
                                            , window_size
                                            , window_size));
    }
    mouse_window.at<Vec3b>((window_size/2), (window_size/2)) = Vec3b(0, 255, 0);
    resize(mouse_window, mouse_window, Size(env_setup.mouse_window_max, env_setup.mouse_window_max), 0, 0, CV_INTER_NN);

    Mat left_capture;
    if(callback_param.click_points_left.size() > 0)
    {
        Point2i left_pt = callback_param.click_points_left[callback_param.click_points_left.size()-1];

        int x = border + left_pt.x;
        int y = border + left_pt.y;
        left_capture = im_left_border(Rect(x-(window_size/2)
                                            , y-(window_size/2)
                                            , window_size
                                            , window_size));

        left_capture.at<Vec3b>((window_size/2), (window_size/2)) = Vec3b(0, 255, 0);
        resize(left_capture, left_capture, Size(env_setup.mouse_window_max, env_setup.mouse_window_max), 0, 0, CV_INTER_NN);
    }
    else
        left_capture = Mat::zeros(Size(env_setup.mouse_window_max, env_setup.mouse_window_max), CV_8UC3);
    
    Mat right_capture;
    if(callback_param.click_points_right.size() > 0)
    {
        Point2i right_pt = callback_param.click_points_right[callback_param.click_points_right.size()-1];

        int x = border + right_pt.x;
        int y = border + right_pt.y;
        right_capture = im_right_border(Rect(x-(window_size/2)
                                            , y-(window_size/2)
                                            , window_size
                                            , window_size));

        right_capture.at<Vec3b>((window_size/2), (window_size/2)) = Vec3b(0, 255, 0);
        resize(right_capture, right_capture, Size(env_setup.mouse_window_max, env_setup.mouse_window_max), 0, 0, CV_INTER_NN);
    }
    else
        right_capture = Mat::zeros(Size(env_setup.mouse_window_max, env_setup.mouse_window_max), CV_8UC3);

    vector<Mat> mouse_window_arr = {mouse_window, left_capture, right_capture};
    Mat mouse_window_result;
    vconcat(mouse_window_arr, mouse_window_result);
    imshow(env_setup.mouse_window_name, mouse_window_result);
}

void calculate_vectors(callback_param_t *cb)
{
    int left_size = cb->click_points_left.size();
    int right_size = cb->click_points_right.size();
    double resize_ratio = cb->resize_ratio;
    
    Rect window_rect = getWindowImageRect(env_setup.window_name);
    int match_size = left_size;
    int im_width = window_rect.width*(1/resize_ratio);
    int im_height = window_rect.height*(1/resize_ratio)/2;

    // convert pixel to radian coordinate, in unit sphere
    // x : longitude
    // y : latitude
    vector<Point2d> key_point_left_radian(match_size);
    vector<Point2d> key_point_right_radian(match_size);
    for(int i = 0; i < match_size; i++)
    {
        key_point_left_radian[i].x = 2*M_PI*(cb->click_points_left[i].x / im_width);
        key_point_right_radian[i].x = 2*M_PI*(cb->click_points_right[i].x / im_width);
        key_point_left_radian[i].y = M_PI*(cb->click_points_left[i].y / im_height);
        key_point_right_radian[i].y = M_PI*(cb->click_points_right[i].y / im_height);
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
    
    cb->eight_point_calculated = 1;
    cb->eight_point_R = result_R;
    cb->eight_point_T = result_T;
}

void try_calulate_vectors(callback_param_t *cb)
{
    static Vec3d R, T;
    
    int left_size = cb->click_points_left.size();
    int right_size = cb->click_points_right.size();
    if((left_size == right_size) && (left_size >= 8))
    {
        calculate_vectors(cb);
        
        stringstream sstrm[5];
        sstrm[0] << "Eight Point Result";
        sstrm[1] << fixed;
        sstrm[1].precision(3);
        sstrm[1] << "Result R vector(degree) : " << cb->eight_point_R*180.0f/M_PI;
        sstrm[2] << fixed;
        sstrm[2].precision(3);
        sstrm[2] << "Result_T vector(unit vector) : " << cb->eight_point_T;
        sstrm[3] << "Press ESC will save rectified image and exit program";
        
        // 8포인트 이상일 때는 이전 벡터 값과 비교
        if(left_size > 8)
        {
            Vec3d *R_ = &(cb->eight_point_R), *T_ = &(cb->eight_point_T);
            double R_diff = abs((*R_)[0] - R[0]) + abs((*R_)[1] - R[1]) + abs((*R_)[2] - R[2]);
            double T_diff = abs((*T_)[0] - T[0]) + abs((*T_)[1] - T[1]) + abs((*T_)[2] - T[2]);
            sstrm[4] << fixed << setprecision(3);
            sstrm[4] << "Vector difference: R=" << R_diff << ", T=" << T_diff;
        }
        else
        {
            sstrm[4] << "";
        }
        vector<string> debug_msg = {sstrm[0].str(), sstrm[1].str(), sstrm[2].str(), sstrm[3].str(), sstrm[4].str()};
        debug_show(debug_msg);
        
        R = cb->eight_point_R;
        T = cb->eight_point_T;
    }
}

static void pick_point(int evt, int x, int y, int flags, void* param)
{
    callback_param_t* callback_param = (callback_param_t*)param;
    double resize_ratio = callback_param->resize_ratio;
    if(evt == CV_EVENT_MOUSEMOVE) // 마우스 이동
    {
        callback_param->pt = Point2i(x, y)/resize_ratio;
        callback_param->pt.x += callback_param->x_offset;
        callback_param->pt.y += callback_param->y_offset;
    }
    else if(evt == CV_EVENT_LBUTTONDOWN) // 왼쪽 마우스 클릭
    {
        Rect window_rect = getWindowImageRect(env_setup.window_name);
        if(callback_param->point_state == LEFT_POINT) // left 영상을 누를 차례
        {
            if(y < window_rect.height/2) // 위쪽 이미지(left 영상)을 누른 경우
            {
                Point2i input = callback_param->pt;
                callback_param->click_points_left.push_back(input);
                callback_param->point_state = RIGHT_POINT;
                
                stringstream sstrm[5];
				sstrm[0] << "left point input";
                sstrm[1] << "Point x: " << input.x << " Point y: " << input.y;
                sstrm[2] << "num of left point: " << callback_param->click_points_left.size();
                sstrm[3] << "num of right point: " << callback_param->click_points_right.size();
                sstrm[4] << "Please Select right point";
                vector<string> debug_msg = {sstrm[0].str(), sstrm[1].str(), sstrm[2].str(), sstrm[3].str(), sstrm[4].str()};
                debug_show(debug_msg);
                
                try_calulate_vectors(callback_param);
            }
            else
            {
                stringstream sstrm[2];
                sstrm[0] << "Not this area";
                sstrm[1] << "Please, Select in left image";
                vector<string> debug_msg = {sstrm[0].str(),sstrm[1].str()};
                debug_show(debug_msg);
            }
        }
        else if(callback_param->point_state == RIGHT_POINT) // right 영상을 누를 차례
        {
            if(y < window_rect.height/2)
            {
                stringstream sstrm[2];
                sstrm[0] << "Not this area";
                sstrm[1] << "Please, Select in right image";
                vector<string> debug_msg = {sstrm[0].str(),sstrm[1].str()};
                debug_show(debug_msg);
            }
            else // 아래쪽 이미지(right 영상)을 누른 경우
            {
                Point2i input = Point2i(callback_param->pt.x, callback_param->pt.y - (window_rect.height/2)/resize_ratio);
                callback_param->click_points_right.push_back(input);
                callback_param->point_state = LEFT_POINT;

                stringstream sstrm[5];
                sstrm[0] << "right point input";
                sstrm[1] << "Point x: " << input.x << " Point y: " << input.y;
                sstrm[2] << "num of left point: " << callback_param->click_points_left.size();
                sstrm[3] << "num of right point: " << callback_param->click_points_right.size();
                sstrm[4] << "Please Select left point";
                vector<string> debug_msg = {sstrm[0].str(), sstrm[1].str(), sstrm[2].str(), sstrm[3].str(), sstrm[4].str()};
                debug_show(debug_msg);
                
                try_calulate_vectors(callback_param);
            }
        }
    }
    else if(evt == CV_EVENT_RBUTTONDOWN) // 우측 마우스 클릭
    {
        // do nothing
    }
    else if(evt == CV_EVENT_MOUSEWHEEL) // 마우스 휠
    {
        if(getMouseWheelDelta(flags) > 0)
            callback_param->magnifying_size += 1;
        else
            callback_param->magnifying_size -= 1;
        
        if(callback_param->magnifying_size > env_setup.mouse_window_max)
            callback_param->magnifying_size = env_setup.mouse_window_max;
        else if(callback_param->magnifying_size < env_setup.mouse_window_min)
            callback_param->magnifying_size = env_setup.mouse_window_min;
    }
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

/* Check whether the file exists */
bool is_file_exists(string filePath)
{
    ifstream infile(filePath);
    return infile.good();
}

bool keyevent_esc(int eightpoint_calculated)
{
    if(eightpoint_calculated)
        return false;
    return true;
}

void keyevent_a(callback_param_t *cb)
{
    Point2d *P;
    if(cb->point_state == LEFT_POINT)
    {
        // 이전에 RIGHT였거나, 아무 것도 누르지 않았을 수도 있음.
        if(cb->click_points_right.size() > 0)
        {
            P = &(cb->click_points_right[cb->click_points_right.size()-1]);
            P->x -= 1;
        }
    }
    else if(cb->point_state == RIGHT_POINT)
    {
        if(cb->click_points_left.size() > 0)
        {
            P = &(cb->click_points_left[cb->click_points_left.size()-1]);
            P->x -= 1;
        }
    }
}

void keyevent_s(callback_param_t *cb)
{
    Point2d *P;
    if(cb->point_state == LEFT_POINT)
    {
        // 이전에 RIGHT였거나, 아무 것도 누르지 않았을 수도 있음.
        if(cb->click_points_right.size() > 0)
        {
            P = &(cb->click_points_right[cb->click_points_right.size()-1]);
            P->y += 1;
        }
    }
    else if(cb->point_state == RIGHT_POINT)
    {
        if(cb->click_points_left.size() > 0)
        {
            P = &(cb->click_points_left[cb->click_points_left.size()-1]);
            P->y += 1;
        }
    }
}

void keyevent_d(callback_param_t *cb)
{
    Point2d *P;
    if(cb->point_state == LEFT_POINT)
    {
        // 이전에 RIGHT였거나, 아무 것도 누르지 않았을 수도 있음.
        if(cb->click_points_right.size() > 0)
        {
            P = &(cb->click_points_right[cb->click_points_right.size()-1]);
            P->x += 1;
        }
    }
    else if(cb->point_state == RIGHT_POINT)
    {
        if(cb->click_points_left.size() > 0)
        {
            P = &(cb->click_points_left[cb->click_points_left.size()-1]);
            P->x += 1;
        }
    }
}

void keyevent_w(callback_param_t *cb)
{
    Point2d *P;
    if(cb->point_state == LEFT_POINT)
    {
        // 이전에 RIGHT였거나, 아무 것도 누르지 않았을 수도 있음.
        if(cb->click_points_right.size() > 0)
        {
            P = &(cb->click_points_right[cb->click_points_right.size()-1]);
            P->y -= 1;
        }
    }
    else if(cb->point_state == RIGHT_POINT)
    {
        if(cb->click_points_left.size() > 0)
        {
            P = &(cb->click_points_left[cb->click_points_left.size()-1]);
            P->y -= 1;
        }
    }
}

void get_output_filenames(string input, string &out_horizontal, string &out_vertical)
{
    PathKit path;
    
    string dname = path.dirname(input);
    string bname = path.basename(input);
    string name, ext;
    path.split_ext(bname, name, ext);
    
    stringstream out_horizontal_s, out_vertical_s;
    if(dname.length() > 0) {
        out_horizontal_s << dname << '/' << name << "_rectified.png";
        out_vertical_s << dname << '/' << name << "_rectified_vertical.png";
    }
    else {
        out_horizontal_s << name << "_rectified.png";
        out_vertical_s << name << "_rectified_vertical.png";
    }
    
    out_horizontal = out_horizontal_s.str();
    out_vertical = out_vertical_s.str();
}

string get_output_logname(string input)
{
    PathKit path;
    
    string dname = path.dirname(input);
    string bname = path.basename(input);
    string name, ext;
    path.split_ext(bname, name, ext);
    
    stringstream out;
    if(dname.length() > 0)
        out << dname << '/' << name << "_vector.txt";
    else
        out << name << "_vector.txt";
    
    return out.str();
}

int main(int argc, char* argv[])
{
    INIReader reader("config_file.ini");
    if(reader.ParseError() != 0)
    {
        cout << "please, check config_file.ini" << endl;
        return 0;
    }
    env_setup.im_left_name = reader.Get("config", "im_left_name", "");
    env_setup.im_right_name = reader.Get("config", "im_right_name", "");
    env_setup.resize_input = reader.GetInteger("config", "resize_input", 0);
    env_setup.resize_input_width = reader.GetInteger("config", "resize_input_width", 0); // 입력 이미지의 resize. 이 크기에 따라서 윈도우 사이즈도 바뀔 것 같다.
    env_setup.resize_input_height = reader.GetInteger("config", "resize_input_height", 0);
    env_setup.output_height = reader.GetInteger("config", "output_height", 960);
    env_setup.mouse_offset_max = reader.GetInteger("config", "mouse_offset_max", 3);
    env_setup.mouse_window_max = reader.GetInteger("config", "mouse_window_max", 201); // mouse window의 사이즈 일지도?
    env_setup.mouse_window_min = reader.GetInteger("config", "mouse_window_min", 5);
    env_setup.window_name = reader.Get("config", "window_name", "test_show");
    env_setup.mouse_window_name = reader.Get("config", "mouse_window_name", "magnifying_tool");
    env_setup.debug_window_name = reader.Get("config", "debug_window_name", "debug_window");
    env_setup.debug_window_width = reader.GetInteger("config", "debug_window_width", 800);
    env_setup.debug_window_height = reader.GetInteger("config", "debug_window_height", 200);
    
    // find input image
    string im_left_name = env_setup.im_left_name;
    string im_right_name = env_setup.im_right_name;
    
    // check input image file exists
    if(!is_file_exists(im_left_name))
    {
        cout << "Input left image " << im_left_name << " is not exists." << endl;
        cout << "Please input left image file path > ";
        getline(cin, im_left_name);
    }
    if(!is_file_exists(im_right_name))
    {
        cout << "Input right image " << im_right_name << " is not exists." << endl;
        cout << "Please input right image file path > ";
        getline(cin, im_right_name);
    }
    
    cout << "Input left: " << im_left_name << endl;
    cout << "Input right: " << im_right_name << endl;
    
    // load image
    Mat im_left = imread(im_left_name);
    Mat im_right = imread(im_right_name);
    
    if(env_setup.resize_input == 1)
    {
        resize(im_left, im_left, Size(env_setup.resize_input_width, env_setup.resize_input_height), 0, 0, INTER_CUBIC);
        resize(im_right, im_right, Size(env_setup.resize_input_width, env_setup.resize_input_height), 0, 0, INTER_CUBIC);
    }
    
    callback_param_t callback_param;
    callback_param.resize_ratio = double(env_setup.output_height)/double(im_left.rows*2);
    callback_param.magnifying_size = 10;
    callback_param.point_state = LEFT_POINT;
    callback_param.pt = Point2i(0, 0);
    callback_param.x_offset = 0;
    callback_param.y_offset = 0;
    callback_param.eight_point_calculated = 0;
    callback_param.eight_point_R = Vec3d(0, 0, 0);
    callback_param.eight_point_T = Vec3d(0, 0, 0);
    
    namedWindow(env_setup.window_name);
    setMouseCallback(env_setup.window_name, pick_point, (void*)&callback_param);
    
    namedWindow(env_setup.mouse_window_name);
    createTrackbar("x_offset", env_setup.mouse_window_name, &(callback_param.x_offset), 7);
    setTrackbarMin("x_offset", env_setup.mouse_window_name, -3);
    setTrackbarMax("x_offset", env_setup.mouse_window_name, 3);
    setTrackbarPos("x_offset", env_setup.mouse_window_name, 0);
    createTrackbar("y_offset", env_setup.mouse_window_name, &(callback_param.y_offset), 7);
    setTrackbarMin("y_offset", env_setup.mouse_window_name, -3);
    setTrackbarMax("y_offset", env_setup.mouse_window_name, 3);
    setTrackbarPos("y_offset", env_setup.mouse_window_name, 0);

    stringstream sstrm1[4];
    sstrm1[0] << "Debug Printer Window, exit program: ESC";
    sstrm1[1] << "Please, select one point in left image";
    sstrm1[2] << "Input image width: " << im_left.cols << " height: " << im_left.rows;
    sstrm1[3] << "Input resized: " << env_setup.resize_input;
    vector<string> debug_msg = {sstrm1[0].str(), sstrm1[1].str(), sstrm1[2].str(), sstrm1[3].str()};
    debug_show(debug_msg);
    
    // 마우스로 특징점 입력을 받는 단계
    bool state_feature_point_input = true;
    while(state_feature_point_input)
    {
        Mat im_left_tmp = im_left.clone();
        Mat im_right_tmp = im_right.clone();
        if(callback_param.click_points_left.size() > 0)
        {
            for(int i = 0; i < callback_param.click_points_left.size(); i++)
                circle(im_left_tmp, callback_param.click_points_left[i], 5, Scalar(255, 0, 0), 3);
        }
        if(callback_param.click_points_right.size() > 0)
        {
			for (int i = 0; i < callback_param.click_points_right.size(); i++)
				circle(im_right_tmp, callback_param.click_points_right[i], 5, Scalar(255, 0, 0), 3);
        }
        imshow_resize(im_left_tmp, im_right_tmp);
        mouse_show(im_left, im_right, callback_param);

        int key_input = waitKey(10);
        switch(key_input) {
            case 27: // ESC
                state_feature_point_input = keyevent_esc(callback_param.eight_point_calculated);
                break;
            case 'A':
            case 'a':
                keyevent_a(&callback_param);
                break;
            case 'S':
            case 's':
                keyevent_s(&callback_param);
                break;
            case 'D':
            case 'd':
                keyevent_d(&callback_param);
                break;
            case 'W':
            case 'w':
                keyevent_w(&callback_param);
                break;
        }
    }
    
    // 특징점 계산이 끝나고 결과를 작성하는 단계
    if(callback_param.eight_point_calculated == 1)
    {
        // log file open
        //string log_name = "manual_estimated_extrinsic.txt";
        string log_name = get_output_logname(im_left_name);
        cout << "Create vector file: " << log_name << endl;
        ofstream log;
        log.open(log_name);
        
        stringstream sstrm[1];
        sstrm[0] << "Save Rectified images";
        vector<string> debug_msg = {sstrm[0].str()};
        debug_show(debug_msg);

        log << "initial_R_vector: " << DEGREE(callback_param.eight_point_R) << endl;
        log << "initial_T_vector: " << callback_param.eight_point_T << endl;

        Mat rectified_left, rectified_right;
        rectify(im_left, im_right, callback_param.eight_point_R, callback_param.eight_point_T, rectified_left, rectified_right);

        erp_rotation erp_rot;
        Mat rot_mat_90deg = erp_rot.eular2rot(Vec3d(RAD(89.999), 0, 0)).inv();
        Mat left_rotate = erp_rot.rotate_image(rectified_left, rot_mat_90deg);
        Mat right_rotate = erp_rot.rotate_image(rectified_right, rot_mat_90deg);
        cv::rotate(left_rotate, left_rotate, cv::ROTATE_90_CLOCKWISE);
        cv::rotate(right_rotate, right_rotate, cv::ROTATE_90_CLOCKWISE);
        
        string file_LH, file_RH, file_LV, file_RV;
        get_output_filenames(im_left_name, file_LH, file_LV);
        get_output_filenames(im_right_name, file_RH, file_RV);
        cout << "Create output left: " << file_LH << endl;
        cout << "Create output right: " << file_RH << endl;
        cout << "Create output left-vertical: " << file_LV << endl;
        cout << "Create output right-vertical: " << file_RV << endl;
        
        //imwrite("rectified_left.png", rectified_left);
        //imwrite("rectified_right.png", rectified_right);
        //imwrite("rectified_left_vertical.png", left_rotate);
        //imwrite("rectified_right_vertical.png", right_rotate);
        imwrite(file_LH, rectified_left);
        imwrite(file_RH, rectified_right);
        imwrite(file_LV, left_rotate);
        imwrite(file_RV, right_rotate);
        
        log.close();
    }

    return 0;
}
