#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Point find_closest_previous_center(Point2f &p, std::vector<Point2f> &prev_centers);
double euclidean_distance(Point2f &p1, Point2f &p2);
int get_largest_contour_idx(std::vector<std::vector<Point> > &contours);

enum direction {upwards, downwards};

int main(int argc, char* argv[]) {
    VideoCapture cap(0);
    std::vector<Point2f> prev_centers;
    direction prev_dir = upwards;
    int throws = 0;
    Mat prev_upper_third;

    while(true) {
        Mat frame, upper_third, frame_delta, processed_frame;
        cap >> frame;

        upper_third = frame(Rect(0, 0, frame.cols, frame.rows/2));

        // Convert to gray
        cvtColor(upper_third, upper_third, COLOR_BGR2GRAY);

        // Reduce the noise so we avoid false circle detection
        GaussianBlur(upper_third, upper_third, Size(9, 9), 2, 2);
        if (prev_upper_third.empty()) {
            prev_upper_third = upper_third;
        } 

        // get difference between frames (i.e. moving balls)
        absdiff(upper_third, prev_upper_third, frame_delta);
        threshold(frame_delta, processed_frame, 25, 255, THRESH_BINARY);

        prev_upper_third = upper_third;

        // erode and dilate to remove impurities
        Mat element = getStructuringElement(MORPH_ELLIPSE, Size(9, 9), Point(-1, -1));
        erode(processed_frame, processed_frame, element, Point(-1, -1), 2);
        dilate(processed_frame, processed_frame, element, Point(-1, -1), 2);

        // find contours
        std::vector<std::vector<Point> > contours;
        Mat processed_frame_copy = processed_frame.clone();  // prevent over-writing original data
        findContours(processed_frame_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        if (contours.size() == 0) {
            prev_centers.clear();
            prev_dir = upwards;
            imshow("Frame", frame);
            if (waitKey(10) == 27) break;
            continue;
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
        // Point2f center = centers[i];
        // Scalar color(rand()&255, rand()&255, rand()&255);
        // drawContours(drawing, contours, i, color);
        
        // circle center
        circle(frame, center, 3, Scalar(0,255,0), -1, 8, 0);
        // circle outline
        circle(frame, center, radius, Scalar(0,0,255), 3, 8, 0);

        direction dir;
        if (prev_centers.empty()) {
            prev_centers.push_back(center);
        } else {
            Point prev_center = find_closest_previous_center(center, prev_centers);

            // row-major order
            if (prev_center.y > center.y) {  // upwards
                dir = upwards;
            } else {  // downwards
                dir = downwards;
            }

            if (prev_dir == upwards && dir == downwards) {
                throws += 1;
                std::cout << throws << std::endl;
            }
            prev_dir = dir;
            
        }
        imshow("Frame", frame);
        if (waitKey(10) == 27) break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
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
