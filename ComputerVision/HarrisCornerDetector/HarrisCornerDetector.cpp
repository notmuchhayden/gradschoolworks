// HarrisCornerDetector.cpp : 이 파일에는 'main' 함수가 포함되어 있습니다. 프로그램 실행이 여기서 시작하고 끝납니다.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
    // 이미지 불러오기
    cv::Mat src = cv::imread("input.png");
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다!" << std::endl;
        return -1;
    }

    // 그레이스케일로 변환하고 float으로 변환
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    cv::Mat fgray;
    gray.convertTo(fgray, CV_32F, 1.0/255.0);

    // 매개변수
    int blockSize = 3;      // 공분산을 위한 이웃 크기 (가우시안에 의해 간접적으로 사용됨)
    int aperture = 3;       // Sobel aperture
    double k = 0.04;        // Harris detector 자유 매개변수
    int gaussianSize = 7;   // 구조 텐서를 위한 평활화 윈도우 크기
    double gaussianSigma = 2.0;

    // 1) 영상 기울기 Ix, Iy 계산
    cv::Mat Ix, Iy;
    cv::Sobel(fgray, Ix, CV_32F, 1, 0, aperture);
    cv::Sobel(fgray, Iy, CV_32F, 0, 1, aperture);

    // 2) 모든 픽셀에서 미분 곱 계산: Ixx, Iyy, Ixy
    cv::Mat Ixx = Ix.mul(Ix);
    cv::Mat Iyy = Iy.mul(Iy);
    cv::Mat Ixy = Ix.mul(Iy);

    // 3) 곱의 합을 얻기 위해 가우시안 필터 적용 (구조 텐서 성분)
    cv::Mat Sxx, Syy, Sxy;
    cv::GaussianBlur(Ixx, Sxx, cv::Size(gaussianSize, gaussianSize), gaussianSigma);
    cv::GaussianBlur(Iyy, Syy, cv::Size(gaussianSize, gaussianSize), gaussianSigma);
    cv::GaussianBlur(Ixy, Sxy, cv::Size(gaussianSize, gaussianSize), gaussianSigma);

    // 4) 각 픽셀에 대해 Harris 응답 R = det(M) - k * trace(M)^2 계산
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

    // 5) 임계값 및 비최대 억제 (3x3 이웃)
    double minVal, maxVal;
    cv::minMaxLoc(response, &minVal, &maxVal);
    // 최대 응답에 대한 상대 임계값
    float thresh = static_cast<float>(0.01 * maxVal);

    cv::Mat corners = cv::Mat::zeros(response.size(), CV_8U);
    int neighborhood = 1; // NMS를 위한 반경 (1 -> 3x3)

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

    // 6) 결과 이미지에 코너 그리기
    cv::Mat result = src.clone();
    for (int y = 0; y < corners.rows; ++y) {
        const uchar* crow = corners.ptr<uchar>(y);
        for (int x = 0; x < corners.cols; ++x) {
            if (crow[x]) {
                cv::circle(result, cv::Point(x, y), 4, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
            }
        }
    }

    // 중간 및 최종 결과 표시
    cv::imshow("Gray", gray);
    cv::imshow("Harris Response (normalized)", response / static_cast<float>(maxVal));
    cv::imshow("Corners", corners);
    cv::imshow("Detected Corners", result);
    cv::imwrite("harris_output.png", result);
    cv::waitKey(0);

    return 0;
}

