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
std::vector<std::vector<Point> > sort_contours_by_area(std::vector<std::vector<Point> > &contours);
void remove_nearby_centers_and_radii(std::vector<Point2f> &centers, std::vector<float> &radii);
void update_balls(std::vector<ball> &balls, std::vector<Point2f> &centers, std::vector<float> &radii);
void update_throws(std::vector<ball> &balls, int &throws);
void update_ball_center(ball &ball, Point2f &center);
direction get_direction(std::deque<Point2f> &center_buffer);

int main(int argc, char* argv[]) {
    int throws = 0;
    std::vector<ball> balls;
    Mat frame, upper_third, prev_upper_third;
    auto start = std::chrono::steady_clock::now();

    VideoCapture cap(0);

    while(true) {
        cap >> frame;

        upper_third = frame(Rect(0, 0, frame.cols, frame.rows/2));

        std::vector<std::vector<Point> > contours = get_contours(upper_third, prev_upper_third);
        // sort contours by area
        contours = sort_contours_by_area(contours);

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

        std::vector<std::vector<Point> > contours_poly(contours.size());
        std::vector<Point2f> centers(contours.size());
        std::vector<float> radii(contours.size());

        for (int i = 0; i < contours.size(); ++i) {
            approxPolyDP(contours[i], contours_poly[i], 3, true);
            minEnclosingCircle(contours_poly[i], centers[i], radii[i]);
        }

        if (contours.size() > 1) {
            remove_nearby_centers_and_radii(centers, radii);
        }
        update_balls(balls, centers, radii);
        update_throws(balls, throws);

        for (int i = 0; i < balls.size(); ++i) {
            // circle center
            circle(frame, balls[i].center_buffer.back(), 3, Scalar(0,255,0), -1, 8, 0);
            // circle outline
            circle(frame, balls[i].center_buffer.back(), balls[i].radius, Scalar(0,0,255), 3, 8, 0);
        }

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

    // imshow("Proccessed", processed_frame);

    // find contours
    std::vector<std::vector<Point> > contours;
    findContours(processed_frame, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    return contours;
}

double euclidean_distance(Point2f &p1, Point2f &p2) {
    double d_x = p1.x - p2.x;
    double d_y = p1.y - p2.y;

    return sqrt(pow(d_x, 2) + pow(d_y, 2));
}

bool sort_desc(const std::pair<double, int> &a, const std::pair<double, int> &b) {
    return a.first > b.first;
}

std::vector<std::vector<Point> > sort_contours_by_area(std::vector<std::vector<Point> > &contours) {
    std::vector<std::pair<double, int> > contour_area_idx_pairs;
    std::vector<std::vector<Point> > return_contours;

    for (int i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area < 10) {  // ignore small contours
            continue;
        }
        contour_area_idx_pairs.push_back(std::make_pair(area, i));
    }

    std::sort(contour_area_idx_pairs.begin(), contour_area_idx_pairs.end(), sort_desc);

    for (int i = 0; i < contours.size(); ++i) {
        // std::cout << contour_area_idx_pairs[i].first << " " << contour_area_idx_pairs[i].second << "\n";
        return_contours.push_back(contours[contour_area_idx_pairs[i].second]);
    }

    return return_contours;
}

void remove_nearby_centers_and_radii(std::vector<Point2f> &centers, std::vector<float> &radii) {
    for (int i = 0; i < centers.size()-1; ++i) {
        int original_center_size = centers.size();
        for (int j = i+1; j < centers.size(); ++j) {
            double dist = euclidean_distance(centers[i], centers[j]);
            if (dist < 100) {
                centers.erase(centers.begin() + j);
                radii.erase(radii.begin() + j);
                --j; // since we remove this index, the element originally further along will now fall back into this place
            }
        }
    }
}

void update_balls(std::vector<ball> &balls, std::vector<Point2f> &centers, std::vector<float> &radii) {
    if (balls.size() == 0) {
        for (int i = 0; i < centers.size(); ++i) {
            ball new_ball(centers[i], radii[i]);
            balls.push_back(new_ball);
        }
    } else {
        if (balls.size() >= centers.size()) {
            // for each center, find the closest ball and retain it
            // remove all leftover balls
            std::vector<ball> balls_to_keep;

            for (int i = 0; i < centers.size(); ++i) {
                double min_dist = std::numeric_limits<double>::max();
                int ball_idx;
                for (int j = 0; j < balls.size(); ++j) {
                    Point2f ball_center = balls[j].center_buffer.back();
                    double dist = euclidean_distance(ball_center, centers[i]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        ball_idx = j;
                    }
                }

                // update ball center
                balls[ball_idx].update_center(centers[i]);

                // retain ball
                balls_to_keep.push_back(balls[ball_idx]);
            }
            balls = balls_to_keep;
        } else {
            // for each ball, find closest center and remove it
            // create new balls for each leftover center
            for (int i = 0; i < balls.size(); ++i) {
                double min_dist = std::numeric_limits<double>::max();
                int center_idx;
                Point2f ball_center = balls[i].center_buffer.back();
                for (int j = 0; j < centers.size(); ++j) {
                    double dist = euclidean_distance(ball_center, centers[j]);
                    if (dist < min_dist) {
                        min_dist = dist;
                        center_idx = j;
                    }
                }

                // update ball center
                balls[i].update_center(centers[center_idx]);

                // remove center and radius
                centers.erase(centers.begin() + center_idx);
                radii.erase(radii.begin() + center_idx);
            }

            // leftover centers require new balls
            for (int i = 0; i < centers.size(); ++i) { 
                ball new_ball(centers[i], radii[i]);
                balls.push_back(new_ball);
            }
        }
    }
}

void update_throws(std::vector<ball> &balls, int &throws) {
    for (int i = 0; i < balls.size(); ++i) {
        if (balls[i].center_buffer.size() <= 1) return;
        balls[i].dir = get_direction(balls[i].center_buffer);
        if (balls[i].prev_dir == upwards && balls[i].dir == downwards) {
            throws += 1;
            std::cout << throws << std::endl;
        }
        balls[i].prev_dir = balls[i].dir;
    }
}

direction get_direction(std::deque<Point2f> &center_buffer) {
    int center_buffer_size = center_buffer.size();  // will be smaller initially

    Point2f past_point = center_buffer[0]; 
    Point2f current_point = center_buffer[center_buffer_size - 1];

    // row-major order
    if (past_point.y > current_point.y) {
        return upwards;
    } else {
        return downwards;
    }
}
