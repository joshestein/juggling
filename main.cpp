#include <chrono>
#include <deque>
#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#define MAX_CENTER_BUFFER 3
#define IDLE_REST_TIME 1.5

using namespace cv;

enum direction {upwards, downwards};

struct ball {
    std::deque<Point2f> center_buffer;
    float radius;
    direction dir;
    direction prev_dir;

    ball(Point2f center, float p_radius) {
        center_buffer.push_back(center);
        radius = p_radius;
        dir = upwards;
        prev_dir = upwards;
    }

    void update_center(Point2f &center) {
        if (center_buffer.size() > MAX_CENTER_BUFFER) {
            center_buffer.pop_front();
        }
        center_buffer.push_back(center);
    }
};

std::vector<std::vector<Point> > get_contours(Mat &upper_third, Mat &prev_upper_third);
double euclidean_distance(Point2f &p1, Point2f &p2);
int get_largest_contour_idx(std::vector<std::vector<Point> > &contours);
direction get_direction(std::deque<Point2f> &center_buffer);

int main(int argc, char* argv[]) {
    VideoCapture cap(0);

    std::deque<Point2f> center_buffer;
    direction prev_dir;
    int throws = 0;
    Mat frame, upper_third, prev_upper_third;
    auto start = std::chrono::steady_clock::now();

    while(true) {
        cap >> frame;

        upper_third = frame(Rect(0, 0, frame.cols, frame.rows/2));

        std::vector<std::vector<Point> > contours = get_contours(upper_third, prev_upper_third);
        prev_upper_third = upper_third;

        if (contours.size() == 0) {
            auto end = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed_seconds = end - start;
            if (elapsed_seconds.count() > IDLE_REST_TIME) {
                std::cout << "\n----------------------------------\n";
                std::cout << "Reseting throws";
                std::cout << "\n----------------------------------\n";
                throws = 0;
                start = std::chrono::steady_clock::now();
            }
            imshow("Frame", frame);
            if (waitKey(10) == 27) break;
            continue;
        } else {
            start = std::chrono::steady_clock::now();
        }

        // find largest circle
        int contour_idx = get_largest_contour_idx(contours);

        // std::vector<std::vector<Point> > contours_poly(contours.size());
        // std::vector<Point2f> centers(contours.size());
        // std::vector<float> radius(contours.size());

        std::vector<Point> contours_poly;
        Point2f center;
        float radius;

        approxPolyDP(contours[contour_idx], contours_poly, 3, true);
        minEnclosingCircle(contours_poly, center, radius);

        // ensure we only keep <CUMULATIVE_CENTER> elements
        if (center_buffer.size() > MAX_CENTER_BUFFER) {
            center_buffer.pop_front();
        }
        center_buffer.push_back(center);

        // Point2f center = centers[i];
        // Scalar color(rand()&255, rand()&255, rand()&255);
        // drawContours(drawing, contours, i, color);
        
        // circle center
        circle(frame, center, 3, Scalar(0,255,0), -1, 8, 0);
        // circle outline
        circle(frame, center, radius, Scalar(0,0,255), 3, 8, 0);

        direction dir = get_direction(center_buffer);
        if (prev_dir == upwards && dir == downwards) {
            throws += 1;
            std::cout << throws << std::endl;
        }
        prev_dir = dir;

        imshow("Frame", frame);
        if (waitKey(10) == 27) break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
}

std::vector<std::vector<Point> > get_contours(Mat &upper_third, Mat &prev_upper_third) {
    Mat frame_delta, processed_frame;

    // Convert to gray
    cvtColor(upper_third, upper_third, COLOR_BGR2GRAY);
    if (prev_upper_third.empty()) {
        prev_upper_third = upper_third;
    } 

    // Reduce the noise so we avoid false circle detection
    GaussianBlur(upper_third, upper_third, Size(9, 9), 2, 2);

    // get difference between frames (i.e. moving balls)
    absdiff(upper_third, prev_upper_third, frame_delta);
    threshold(frame_delta, processed_frame, 25, 255, THRESH_BINARY);

    // erode and dilate to remove impurities
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(9, 9), Point(-1, -1));
    erode(processed_frame, processed_frame, element, Point(-1, -1), 2);
    dilate(processed_frame, processed_frame, element, Point(-1, -1), 2);

    // find contours
    std::vector<std::vector<Point> > contours;
    findContours(processed_frame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}

Point find_closest_previous_center(Point2f &p, std::vector<Point2f> &prev_centers) {
    double min_dist = std::numeric_limits<double>::max();
    Point closest_point;
    for (auto iter = prev_centers.begin(); iter != prev_centers.end(); ++iter) {
        double dist = euclidean_distance(p, *iter);
        if (dist < min_dist) {
            min_dist = dist;
            closest_point = *iter;
        }
    }
    return closest_point;
}

double euclidean_distance(Point2f &p1, Point2f &p2) {
    double d_x = p1.x - p2.x;
    double d_y = p1.y - p2.y;

    return sqrt(pow(d_x, 2) + pow(d_y, 2));
}

int get_largest_contour_idx(std::vector<std::vector<Point> > &contours) {
    double largest_area = 0.0;
    int contour_idx = 0;
    for (int i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > largest_area) {
            largest_area = area;
            contour_idx = i;
        } 
    }
    return contour_idx;
}

direction get_direction(std::deque<Point2f> &center_buffer) {
    int center_buffer_size = center_buffer.size();  // will be smaller initially

    Point2f past_point = center_buffer[center_buffer_size - 1]; 
    Point2f current_point = center_buffer[0];

    // row-major order
    if (past_point.y > current_point.y) {
        return upwards;
    } else {
        return downwards;
    }
}
