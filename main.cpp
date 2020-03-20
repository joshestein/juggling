#include <iostream>
#include <limits>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

Point find_closest_previous_center(Point &p, std::vector<Point> &prev_centers);
double euclidean_distance(Point &p1, Point &p2);

enum direction {upwards, downwards};

int main(int argc, char* argv[]) {
    VideoCapture cap(0);
    std::vector<Point> prev_centers;
    direction prev_dir;
    int throws = 0;

    while(true) {
        Mat frame, upper_third, upper_third_grey;
        cap >> frame;

        upper_third = frame(Rect(0, 0, frame.cols, frame.rows/2));

        // Convert to gray
        cvtColor(upper_third, upper_third_grey, COLOR_BGR2GRAY);

        // Reduce the noise so we avoid false circle detection
        GaussianBlur(upper_third_grey, upper_third_grey, Size(9, 9), 2, 2);

        std::vector<Vec3f> circles;
        std::vector<Point> centers;
        direction dir;

        HoughCircles(upper_third_grey, circles, HOUGH_GRADIENT, 1, upper_third_grey.rows/8, 100, 22, 0, 0);
        if (circles.size() == 0) {
            centers.clear();
            prev_dir = upwards;
        }
        
        for( size_t i = 0; i < circles.size(); ++i) {
            Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));

            centers.push_back(center);

            if (prev_centers.empty()) {
                prev_centers.push_back(center);
                prev_dir = upwards;
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

            int radius = cvRound(circles[i][2]);
            // circle center
            circle( frame, center, 3, Scalar(0,255,0), -1, 8, 0 );
            // circle outline
            circle( frame, center, radius, Scalar(0,0,255), 3, 8, 0 );
        }

        imshow("Frame", frame);
        if (waitKey(10) == 27) break;
    }
    cap.release();
    destroyAllWindows();

    return 0;
}

Point find_closest_previous_center(Point &p, std::vector<Point> &prev_centers) {
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

double euclidean_distance(Point &p1, Point &p2) {
    double d_x = p1.x - p2.x;
    double d_y = p1.y - p2.y;

    return sqrt(pow(d_x, 2) + pow(d_y, 2));
}
