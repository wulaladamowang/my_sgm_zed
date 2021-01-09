#include <opencv2/imgproc.hpp>
#include <sys/time.h>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>
#include "my_camera.hpp"
#include "get_point_cloud.hpp"
#include <libsgm.h>
#include <string>



struct device_buffer
{
	device_buffer() : data(nullptr) {}
	device_buffer(size_t count) { allocate(count); }
	void allocate(size_t count) { cudaMalloc(&data, count); }
	~device_buffer() { cudaFree(data); }
	void* data;
};
struct timeval t1, t2, t3, t4;
int get_disparity(cv::Mat& img_left, cv::Mat& img_right, cv::Mat& disparity){
     
    static const int disp_size = 128;
    static const int width = img_left.cols;
    static const int height = img_left.rows;
    
    static const int input_depth = img_left.type() == CV_8U ? 8 : 16;
    static const int input_bytes = input_depth * width * height / 8;
    static const int output_depth = disp_size < 256 ? 8 : 16;
    //const int output_depth = 16;
    static const int output_bytes = output_depth * width * height / 8;

    static sgm::StereoSGM sgm(width, height, disp_size, input_depth, output_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);
    static const int invalid_disp = output_depth == 8
                ? static_cast< uint8_t>(sgm.get_invalid_disparity())
                : static_cast<uint16_t>(sgm.get_invalid_disparity());

    // static cv::Mat disparity(height, width, output_depth == 8 ? CV_8U : CV_16U);

    static cv::Mat disparity_8u, disparity_color;
    static device_buffer d_I1(input_bytes), d_I2(input_bytes), d_disparity(output_bytes);


    
    
    cudaMemcpy(d_I1.data, img_left.data, input_bytes, cudaMemcpyHostToDevice);//input_bytes : nBytes
	cudaMemcpy(d_I2.data, img_right.data, input_bytes, cudaMemcpyHostToDevice);
    
    sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(disparity.data, d_disparity.data, output_bytes, cudaMemcpyDeviceToHost);
    // cv::resize(disparity, disparity, cv::Size(disparity.cols*2, disparity.rows*2), cv::INTER_LINEAR);

    // double maxv, minv;
    // cv::minMaxLoc(disparity,&minv,&maxv,0,0);
    // std::cout << "max: " << maxv << " min: " << minv << std::endl;
    // disparity.convertTo(disparity_8u, CV_8U, 255. / 128);// 255/disp_size 比例因子
    // cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
	// disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == invalid_disp);//将非法视差设置为cv::Scalar(0,0,0)
    // cv::imshow("left image", img_left);
    // cv::imshow("disparity" , disparity);
    // cv::imshow("disparity 8u", disparity_8u);
    
    // cv::imshow("disparity color", disparity_color);
    // const char c = cv::waitKey(0);
    // if(c == 's')
    // {
    //     cv::imwrite("img_disparity.jpg", disparity);
    // }
    // cv::destroyAllWindows();
    // if (c == 27) // ESC
    //     break;
    return 0;
}

int relativeDis(cv::Vec4f line_para, std::vector<cv::Point2f> point) {
    double A = line_para[1]/line_para[0];
    double B = -1;
    double C = line_para[3]*(1-line_para[1]/line_para[0]);
    double min = 0.0;
    int index = -1;
    for(int i=0;i<point.size();i++){
        double dis = A*point[i].x+B*point[i].y+C;
        if(dis<0)
            dis = -dis;
        if(-1==index || dis<min)
        {
            min = dis;
            index = i;
        }
    }
    return index;
};

void get_roi(cv::Mat& image, cv::Mat& mask, bool& has_roi, std::vector<int>& rect_roi) {
    
    static cv::Size mask_size = image.size();
    static const cv::Ptr<cv::aruco::Dictionary> c_dictionary = cv::aruco::getPredefinedDictionary(
        cv::aruco::DICT_4X4_50);//DICT_6X6_1000
    static const double pi = acos(-1.0);
    static std::vector<std::vector<cv::Point2f>> marker_corners;
    static std::vector<int> marker_ids;
    int min_x, min_y, max_x, max_y;
    cv::aruco::detectMarkers(image, c_dictionary, marker_corners, marker_ids);
    ///获得检测的面积最大的aruco marker序号,每个ID都有一个（若检测到）
    int marker_number = marker_ids.size();
    if(0 != marker_number)
    {
        int buff_id[6] = {-1, -1, -1, -1, -1, -1};//若存在相应的aruco marker id, 则记录其序号
        double buff_id_length[6] = {0.0, 0, 0, 0, 0, 0};//若存在相应的aruco marker， 则记录其在当前时刻周长的最大值
        for(int i=0;i<marker_number;i++){
            double cur_marker_len = cv::arcLength(marker_corners[i], true);
            if(1 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[1]){
                    buff_id_length[1] = cur_marker_len;
                    buff_id[1] = i;
                }
            }else if(2 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[2]){
                    buff_id_length[2] = cur_marker_len;
                    buff_id[2] = i;
                }
            }else if(3 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[3]){
                    buff_id_length[3] = cur_marker_len;
                    buff_id[3] = i;
                }
            }else if(4 == marker_ids[i]){
                if(cur_marker_len > buff_id_length[4]){
                    buff_id_length[4] = cur_marker_len;
                    buff_id[4] = i;
                }
           }else if(5 == marker_ids[i]){
               if(cur_marker_len > buff_id_length[5]){
                   buff_id_length[5] = cur_marker_len;
                   buff_id[5] = i;
               }
            } else
                continue;
        }

        std::vector<cv::Point2f> compute_line;///用于选中ID的线，每个存储的为选中的ID的中点坐标
        std::vector<int> no_id;///存储被选中的ID的id号
        for(int m=5;m>0;m--){
            if(-1!=buff_id[m]){
                compute_line.emplace_back(cv::Point2f((marker_corners[buff_id[m]][1].x + marker_corners[buff_id[m]][2].x + marker_corners[buff_id[m]][3].x + marker_corners[buff_id[m]][0].x)/4,
                                                      (marker_corners[buff_id[m]][1].y + marker_corners[buff_id[m]][2].y + marker_corners[buff_id[m]][3].y + marker_corners[buff_id[m]][0].y)/4));
                no_id.push_back(m);
            }
        }
        int center_x ;
        int center_y ;
        int index = 0;///记录compute_line中距离拟合线最近的点的序号
        if(no_id.size()<3){
        }else{
            cv::Vec4f line_para;
            cv::fitLine(compute_line, line_para, cv::DIST_L2, 0, 1e-2, 1e-2);
            index = relativeDis(line_para, compute_line);
        }
        center_x = compute_line[index].x;
        center_y = compute_line[index].y;

        std::vector<cv::Point > roi_position;//用于记录roi四个角点的位置
        roi_position.reserve(4);
        cv::Point roi_p0, roi_p1, roi_p2, roi_p3;//用来包裹整个目标圆柱
        ///轴向比例扩大，该参数通过过分支choose coefficient 获得，与目标检测物与贴码位置有关
        float axial_coefficient = 0;//目标长度为marker边长的倍数
        float axial_position_coefficient = 0.0;//marker的ID不同，则则其位于目标的位置不同，距离ID1的上端的位置分数比例
        float radial_coefficient = 0;//目标横向（径向）为marker边长的倍数
        switch (no_id[index]) {
            case 5 : axial_coefficient = 12 ; axial_position_coefficient = 0.13 ; radial_coefficient = 1.4  ;break;
            case 4 : axial_coefficient = 13 ; axial_position_coefficient = 0.32 ; radial_coefficient = 1.9  ;break;
            case 3 : axial_coefficient = 14 ; axial_position_coefficient = 0.50 ; radial_coefficient = 2.2 ;break;
            case 2 : axial_coefficient = 16 ; axial_position_coefficient = 0.67 ; radial_coefficient = 2.5 ;break;
            case 1 : axial_coefficient = 17 ; axial_position_coefficient = 0.84 ; radial_coefficient = 3.1 ;break;
        }

        int i = no_id[index];
        ///纵向比例扩大

        roi_p0.x = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][1].x - marker_corners[buff_id[i]][0].x)+marker_corners[buff_id[i]][0].x;
        roi_p0.y = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][1].y - marker_corners[buff_id[i]][0].y)+marker_corners[buff_id[i]][0].y;

        roi_p1.x = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][0].x - marker_corners[buff_id[i]][1].x)+marker_corners[buff_id[i]][1].x;
        roi_p1.y = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][0].y - marker_corners[buff_id[i]][1].y)+marker_corners[buff_id[i]][1].y;

        roi_p2.x = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][3].x - marker_corners[buff_id[i]][2].x)+marker_corners[buff_id[i]][2].x;
        roi_p2.y = (1-axial_position_coefficient)*axial_coefficient*(marker_corners[buff_id[i]][3].y - marker_corners[buff_id[i]][2].y)+marker_corners[buff_id[i]][2].y;

        roi_p3.x = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][2].x - marker_corners[buff_id[i]][3].x)+marker_corners[buff_id[i]][3].x;
        roi_p3.y = axial_position_coefficient*axial_coefficient*(marker_corners[buff_id[i]][2].y - marker_corners[buff_id[i]][3].y)+marker_corners[buff_id[i]][3].y;
        ///径向比例扩大

        roi_p0.x = radial_coefficient*(marker_corners[buff_id[i]][0].x - marker_corners[buff_id[i]][3].x)+roi_p0.x;
        roi_p0.y = radial_coefficient*(marker_corners[buff_id[i]][0].y - marker_corners[buff_id[i]][3].y)+roi_p0.y;

        roi_p1.x = radial_coefficient*(marker_corners[buff_id[i]][1].x - marker_corners[buff_id[i]][2].x)+roi_p1.x;
        roi_p1.y = radial_coefficient*(marker_corners[buff_id[i]][1].y - marker_corners[buff_id[i]][2].y)+roi_p1.y;

        roi_p2.x = radial_coefficient*(marker_corners[buff_id[i]][2].x - marker_corners[buff_id[i]][1].x)+roi_p2.x;
        roi_p2.y = radial_coefficient*(marker_corners[buff_id[i]][2].y - marker_corners[buff_id[i]][1].y)+roi_p2.y;

        roi_p3.x = radial_coefficient*(marker_corners[buff_id[i]][3].x - marker_corners[buff_id[i]][0].x)+roi_p3.x;
        roi_p3.y = radial_coefficient*(marker_corners[buff_id[i]][3].y - marker_corners[buff_id[i]][0].y)+roi_p3.y;

        center_x = (roi_p0.x+roi_p1.x+roi_p2.x+roi_p3.x)/4;
        center_y = (roi_p0.y+roi_p1.y+roi_p2.y+roi_p3.y)/4;


        mask.setTo(0);
        roi_position.push_back(roi_p0);
        roi_position.push_back(roi_p1);
        roi_position.push_back(roi_p2);
        roi_position.push_back(roi_p3);


        min_x = roi_p0.x>0?roi_p0.x:0;
        min_y = roi_p0.y>0?roi_p0.y:0;
        max_x = roi_p0.x>mask_size.width-1?mask_size.width-1:roi_p0.x;
        max_y = roi_p0.y>mask_size.height-1?mask_size.height-1:roi_p0.y;

        for(int m = 1;m<4;m++){
            if(roi_position[m].x<min_x)
                min_x = roi_position[m].x>0?roi_position[m].x:0;
            if(roi_position[m].y<min_y)
                min_y = roi_position[m].y>0?roi_position[m].y:0;
            if(roi_position[m].x>max_x)
                max_x = roi_position[m].x>mask_size.width-1?mask_size.width-1:roi_position[m].x;
            if(roi_position[m].y>max_y)
                max_y = roi_position[m].y>mask_size.height-1?mask_size.height-1:roi_position[m].y;
        };
        std::vector<std::vector<cv::Point>> contours;
        contours.push_back(roi_position);
        cv::fillPoly(mask, contours, 1);
        has_roi = true;
        // for(int i=0;i<4;i++){
        //     cv::line(image, roi_position[i%4], roi_position[(i+1)%4], cv::Scalar(255,0,0));
        // }
        // cv::imshow("image", image);
        // cv::waitKey(0);
        //cv::destroyAllWindows();
    }else{
        mask.setTo(0);
        std::cout << "No aruco marker detected " << "\n";
        has_roi = false;
    }
}


int main(int argc, char** argv){
    cv::Mat img_left = cv::imread(argv[1], -1);
    cv::Mat img_right = cv::imread(argv[2], -1);
    int a;
    sscanf(argv[3], "%d", &a);
    cv::Size siz = cv::Size(img_left.cols/a, img_left.rows/a);
    cv::Mat disparity = cv::Mat(siz, CV_8U);
    cv::Mat disparity_8u, disparity_color;
    int i = 0;
    const int invalid_disp = 255;
    for(int i=0;i<2;i++){
        gettimeofday(&t1, nullptr);
        cv::resize(img_left, img_left, siz, cv::INTER_LINEAR);
        cv::resize(img_right, img_right, siz, cv::INTER_LINEAR);
        get_disparity(std::ref(img_left), std::ref(img_right), std::ref(disparity));  
        disparity.convertTo(disparity_8u, CV_8U, 255. / 128);
        cv::applyColorMap(disparity_8u, disparity_color, cv::COLORMAP_JET);
		disparity_color.setTo(cv::Scalar(0, 0, 0), disparity == invalid_disp);
        gettimeofday(&t2, nullptr);
        std::cout << " sgm 花费了： "<<(t2.tv_sec-t1.tv_sec)*1000+(t2.tv_usec-t1.tv_usec)/1000 << "毫秒" << std::endl; 
    }
    bool has_roi = false;
    cv::Mat mask = cv::Mat(img_left.size(), CV_8U);
    std::vector<int> rect_roi;
    gettimeofday(&t3, nullptr);
    get_roi(std::ref(img_left), std::ref(mask), std::ref(has_roi), std::ref(rect_roi));
    cv::Mat dst = disparity.mul(mask);
    gettimeofday(&t4, nullptr);
    std::cout << " process 花费了： "<<(t4.tv_sec-t3.tv_sec)*1000+(t4.tv_usec-t3.tv_usec)/1000 << "毫秒" << std::endl; 


    cv::imshow("mask", mask*255);
    cv::imshow("disparity", disparity_8u);
    cv::imshow("disparity_color", disparity_color);
    cv::imshow("dst", dst);
    cv::imshow("img_left", img_left);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}

// int main(int argc, char** argv) {
//     int nb_detected_zed = 1;
//     std::vector<sl::Camera> zeds;
//     init_camera_parameters(nb_detected_zed, zeds);

//     bool run = true;
//     // Create a grab thread for each opened camera
//     std::vector<std::thread> thread_pool_grab_image(nb_detected_zed); // compute threads
//     std::vector<std::thread> thread_pool_get_point_cloud(nb_detected_zed);//获得点云
//     std::vector<cv::Mat> images_left(nb_detected_zed); // display images
//     std::vector<cv::Mat> images_right(nb_detected_zed); // display images
//     std::vector<std::string> wnd_names_left(nb_detected_zed); // display windows names
//     std::vector<std::string> wnd_names_right(nb_detected_zed); // display windows names
//     std::vector<std::vector<cv::Mat>> maps(nb_detected_zed);
//     std::vector<cv::Mat> disparity_imgs(nb_detected_zed);

//     std::vector<long long> images_ts(nb_detected_zed); // images timestamps

//     std::vector<cv::Mat> images_left_gray(nb_detected_zed);
//     std::vector<cv::Mat> images_right_gray(nb_detected_zed);
//     std::vector<cv::Mat> mats;//临时存储参数矩阵


//     for (int z = 0; z < nb_detected_zed; z++)
//         if (zeds[z].isOpened()) {
//             sl::Resolution res = zeds[z].getCameraInformation().camera_configuration.resolution;
//             const int w_low_res = res.width;
//             const int h_low_res = res.height;
//             get_image_transform_mat(argv[1], argv[2], mats);
//             get_tranform_mat(mats, cv::Size(w_low_res, h_low_res), maps[z]);

//             images_left.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC3));
//             images_right.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC3));
//             images_left_gray.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC1));
//             images_right_gray.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC1));
//             disparity_imgs.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC1));
//             // create an image to store Left+Depth image
//             // camera acquisition thread
//             thread_pool_grab_image[z] = std::thread(zed_acquisition, std::ref(zeds[z]), std::ref(images_left[z]), std::ref(images_right[z]), std::ref(maps[z]), std::ref(run), std::ref(images_ts[z]));
//             thread_pool_get_point_cloud[z] = std::thread(get_disparity, std::ref(images_left[z]), std::ref(images_right[z]), std::ref(images_ts[z]), std::ref(run), std::ref(disparity_imgs[z]));
//             // create windows for display
//             wnd_names_left[z] = "ZED(left) ID: " + std::to_string(z);
//             wnd_names_right[z] = "ZED(right) ID: " + std::to_string(z);
//             cv::namedWindow(wnd_names_left[z]);
//             cv::namedWindow(wnd_names_right[z]);
//         }


//     std::vector<long long> last_ts(nb_detected_zed, 0); // use to detect new images
//     char key = ' ';
//     // Loop until 'Esc' is pressed
//     int i = 6;

//     while (key != 'q') {
//         // Resize and show images
//         // for (int z = 0; z < nb_detected_zed; z++) {
//             if (images_ts[0] > last_ts[0]) { // if the current timestamp is newer it is a new image
//                 last_ts[0] = images_ts[0];
//                 cv::cvtColor(images_left[0], images_left_gray[0], cv::COLOR_RGB2GRAY);
//                 cv::cvtColor(images_right[0], images_right_gray[0], cv::COLOR_RGB2GRAY);
                
//                 cv::imshow(wnd_names_left[0], images_left[0]);
//                 cv::imshow(wnd_names_right[0], images_right[0]);
//             } 
//         //}
//         key = cv::waitKey(10);
//     }

//     // stop all running threads
//     run = false;

//     // Wait for every thread to be stopped
//     for (int z = 0; z < nb_detected_zed; z++)
//         if (zeds[z].isOpened()) {
//             thread_pool_grab_image[z].join();
//             thread_pool_get_point_cloud[z].join();
//             zeds[z].close();
//         }
//     return EXIT_SUCCESS;
// }








