// HarrisCornerDetector.cpp : 이 파일에는 'main' 함수가 포함되어 있습니다. 프로그램 실행이 여기서 시작하고 끝납니다.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

int main()
{
	// 1. 이미지 전처리
    // 이미지 불러오기
    cv::Mat src = cv::imread("../assets/input.png");
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
    int block_size = 3;      // 공분산을 위한 이웃 크기 (가우시안에 의해 간접적으로 사용됨)
    int aperture = 3;       // Sobel aperture
    double k = 0.04;        // Harris detector 자유 매개변수
    int gaussian_size = 7;   // 구조 텐서를 위한 평활화 윈도우 크기
    double gaussian_sigma = 2.0;

    // 2. 영상 기울기 ix, iy 계산
    cv::Mat ix, iy;
    cv::Sobel(fgray, ix, CV_32F, 1, 0, aperture);
    cv::Sobel(fgray, iy, CV_32F, 0, 1, aperture);

    // 3. 모든 픽셀에서 미분 곱 계산: ixx, iyy, ixy
    cv::Mat ixx = ix.mul(ix);
    cv::Mat iyy = iy.mul(iy);
    cv::Mat ixy = ix.mul(iy);

    // 4. 곱의 합을 얻기 위해 가우시안 필터 적용 (구조 텐서 성분)
    cv::Mat sxx, syy, sxy;
    cv::GaussianBlur(ixx, sxx, cv::Size(gaussian_size, gaussian_size), gaussian_sigma);
    cv::GaussianBlur(iyy, syy, cv::Size(gaussian_size, gaussian_size), gaussian_sigma);
    cv::GaussianBlur(ixy, sxy, cv::Size(gaussian_size, gaussian_size), gaussian_sigma);

    // 5. 각 픽셀에 대해 Harris 응답 R = det(M) - k * trace(M)^2 계산
    cv::Mat response = cv::Mat::zeros(fgray.size(), CV_32F);
    for (int y = 0; y < fgray.rows; ++y) {
        float* pres = response.ptr<float>(y);
        const float* psxx = sxx.ptr<float>(y);
        const float* psyy = syy.ptr<float>(y);
        const float* psxy = sxy.ptr<float>(y);
        for (int x = 0; x < fgray.cols; ++x) {
            float a = psxx[x];
            float b = psxy[x];
            float c = psyy[x];
            float det = a * c - b * b;
            float trace = a + c;
            pres[x] = det - static_cast<float>(k * trace * trace);
        }
    }

    // 6. 임계값 및 비최대 억제 (3x3 이웃)
    double min_val, max_val;
    cv::minMaxLoc(response, &min_val, &max_val);
    // 최대 응답에 대한 상대 임계값
    float thresh = static_cast<float>(0.01 * max_val);

    cv::Mat corners = cv::Mat::zeros(response.size(), CV_8U);
    int nsize = 1; // NMS를 위한 반경 (1 -> 3x3)

    for (int y = nsize; y < response.rows - nsize; ++y) {
        for (int x = nsize; x < response.cols - nsize; ++x) {
            float val = response.at<float>(y, x);
            if (val <= thresh) 
                continue;

            bool is_max = true;
            for (int dy = -nsize; dy <= nsize && is_max; ++dy) {
                for (int dx = -nsize; dx <= nsize; ++dx) {
                    if (dy == 0 && dx == 0) 
                        continue;

                    if (response.at<float>(y + dy, x + dx) > val) { 
                        is_max = false; break; 
                    }
                }
            }
            if (is_max) 
                corners.at<uchar>(y, x) = 255;
        }
    }

    // 7. 결과 이미지에 코너 그리기
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
    //cv::imshow("Gray", gray);
    //cv::imshow("Harris Response (normalized)", response / static_cast<float>(max_val));
    //cv::imshow("Corners", corners);
    cv::imshow("Detected Corners", result);
    cv::imwrite("../assets/harris_output.png", result);
    cv::waitKey(0);

    return 0;
}

