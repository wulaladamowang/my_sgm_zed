#include <opencv2/imgproc.hpp>
#include <sys/time.h>
#include "my_camera.hpp"


int main(int argc, char** argv) {
    int nb_detected_zed = 1;
    std::vector<sl::Camera> zeds;
    init_camera_parameters(nb_detected_zed, zeds);

    bool run = true;
    // Create a grab thread for each opened camera
    std::vector<std::thread> thread_pool(nb_detected_zed); // compute threads
    std::vector<cv::Mat> images_left(nb_detected_zed); // display images
    std::vector<cv::Mat> images_right(nb_detected_zed); // display images
    std::vector<std::string> wnd_names_left(nb_detected_zed); // display windows names
    std::vector<std::string> wnd_names_right(nb_detected_zed); // display windows names
    std::vector<sl::Timestamp> images_ts(nb_detected_zed); // images timestamps
    std::vector<cv::Mat> images_left_gray(nb_detected_zed);
    std::vector<cv::Mat> images_right_gray(nb_detected_zed);

    for (int z = 0; z < nb_detected_zed; z++)
        if (zeds[z].isOpened()) {
            sl::Resolution res = zeds[z].getCameraInformation().camera_configuration.resolution;
            const int w_low_res = res.width;
            const int h_low_res = res.height;
            images_left.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC3));
            images_right.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC3));
            images_left_gray.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC1));
            images_right_gray.emplace_back(cv::Mat(h_low_res, w_low_res, CV_8UC1));
            // create an image to store Left+Depth image
            // camera acquisition thread
            thread_pool[z] = std::thread(zed_acquisition, std::ref(zeds[z]), std::ref(images_left[z]), std::ref(images_right[z]), std::ref(run), std::ref(images_ts[z]));
            // create windows for display
            wnd_names_left[z] = "ZED(left) ID: " + std::to_string(z);
            wnd_names_right[z] = "ZED(right) ID: " + std::to_string(z);
            cv::namedWindow(wnd_names_left[z]);
            cv::namedWindow(wnd_names_right[z]);
        }


    std::vector<sl::Timestamp> last_ts(nb_detected_zed, 0); // use to detect new images
    char key = ' ';
    // Loop until 'Esc' is pressed
    int i = 6;

    while (key != 'q') {
        // Resize and show images
        // for (int z = 0; z < nb_detected_zed; z++) {
            if (images_ts[0] > last_ts[0]) { // if the current timestamp is newer it is a new image
                last_ts[0] = images_ts[0];
                cv::cvtColor(images_left[0], images_left_gray[0], cv::COLOR_RGB2GRAY);
                cv::cvtColor(images_right[0], images_right_gray[0], cv::COLOR_RGB2GRAY);
                
                cv::imshow(wnd_names_left[0], images_left[0]);
                cv::imshow(wnd_names_right[0], images_right[0]);
            }
            
            if('s' == key){
                i++;
                std::cout << i << std::endl;
                std::string left = "/home/wh/图片/pic/left/left_" + std::to_string(i) + ".jpg";
                std::string right = "/home/wh/图片/pic/right/right_" + std::to_string(i) + ".jpg";
                cv::imwrite(left, images_left[0]);
                cv::imwrite(right, images_right[0]);
                key = ' ';
            }
            
        //}
        key = cv::waitKey(10);
    }

    // stop all running threads
    run = false;

    // Wait for every thread to be stopped
    for (int z = 0; z < nb_detected_zed; z++)
        if (zeds[z].isOpened()) {
            thread_pool[z].join();
            zeds[z].close();
        }
    return EXIT_SUCCESS;
}







