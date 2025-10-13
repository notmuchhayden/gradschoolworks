// HarrisCornerDetector.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
    // Load image
    cv::Mat src = cv::imread("input.png");
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다!" << std::endl;
        return -1;
    }

    // Convert to grayscale and to float
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat fgray;
    gray.convertTo(fgray, CV_32F, 1.0/255.0);

    // Parameters
    int blockSize = 3;      // neighborhood size for covariance (used implicitly via Gaussian)
    int aperture = 3;       // Sobel aperture
    double k = 0.04;        // Harris detector free parameter
    int gaussianSize = 7;   // smoothing window for structure tensor
    double gaussianSigma = 2.0;

    // 1) Compute image gradients Ix, Iy
    cv::Mat Ix, Iy;
    cv::Sobel(fgray, Ix, CV_32F, 1, 0, aperture);
    cv::Sobel(fgray, Iy, CV_32F, 0, 1, aperture);

    // 2) Compute products of derivatives at every pixel: Ixx, Iyy, Ixy
    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // 3) Apply Gaussian filter to the products to obtain sums of products (structure tensor components)
    cv::Mat Sxx, Syy, Sxy;
    cv::GaussianBlur(Ixx, Sxx, cv::Size(gaussianSize, gaussianSize), gaussianSigma);
    cv::GaussianBlur(Iyy, Syy, cv::Size(gaussianSize, gaussianSize), gaussianSigma);
    cv::GaussianBlur(Ixy, Sxy, cv::Size(gaussianSize, gaussianSize), gaussianSigma);

    // 4) Compute Harris response R = det(M) - k * trace(M)^2 per pixel
    cv::Mat response = cv::Mat::zeros(fgray.size(), CV_32F);
    for (int y = 0; y < fgray.rows; ++y) {
        float* rptr = response.ptr<float>(y);
        const float* sxx = Sxx.ptr<float>(y);
        const float* syy = Syy.ptr<float>(y);
        const float* sxy = Sxy.ptr<float>(y);
        for (int x = 0; x < fgray.cols; ++x) {
            float a = sxx[x];
            float b = sxy[x];
            float c = syy[x];
            float det = a * c - b * b;
            float trace = a + c;
            rptr[x] = det - static_cast<float>(k * trace * trace);
        }
    }

    // 5) Threshold and non-maximum suppression (3x3 neighborhood)
    double minVal, maxVal;
    cv::minMaxLoc(response, &minVal, &maxVal);
    // threshold relative to maximum response
    float thresh = static_cast<float>(0.01 * maxVal);

    cv::Mat corners = cv::Mat::zeros(response.size(), CV_8U);
    int neighborhood = 1; // radius for NMS (1 -> 3x3)

    for (int y = neighborhood; y < response.rows - neighborhood; ++y) {
        for (int x = neighborhood; x < response.cols - neighborhood; ++x) {
            float val = response.at<float>(y, x);
            if (val <= thresh) continue;
            bool isMax = true;
            for (int dy = -neighborhood; dy <= neighborhood && isMax; ++dy) {
                for (int dx = -neighborhood; dx <= neighborhood; ++dx) {
                    if (dy == 0 && dx == 0) continue;
                    if (response.at<float>(y + dy, x + dx) > val) { isMax = false; break; }
                }
            }
            if (isMax) corners.at<uchar>(y, x) = 255;
        }
    }

    // 6) Draw corners on result image
    cv::Mat result = src.clone();
    for (int y = 0; y < corners.rows; ++y) {
        const uchar* crow = corners.ptr<uchar>(y);
        for (int x = 0; x < corners.cols; ++x) {
            if (crow[x]) {
                cv::circle(result, cv::Point(x, y), 4, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
        }
    }

    // Show intermediate and final results
    cv::imshow("Gray", gray);
    cv::imshow("Harris Response (normalized)", response / static_cast<float>(maxVal));
    cv::imshow("Corners", corners);
    cv::imshow("Detected Corners", result);
    cv::imwrite("harris_output.png", result);
    cv::waitKey(0);

    return 0;
}

