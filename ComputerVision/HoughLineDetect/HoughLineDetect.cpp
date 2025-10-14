// HoughLineDetect.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>


int main()
{
    // 1. 이미지 전처리
    // 이미지 불러오기
    cv::Mat src = cv::imread("input.png");
    if (src.empty()) {
        std::cerr << "이미지를 불러올 수 없습니다!" << std::endl;
        return -1;
    }

    // 그레이스케일 변환
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);

    // 가우시안 블러 적용 (노이즈 제거)
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

    // 2. 에지 검출 (Canny 알고리즘)
    cv::Mat edges;
    cv::Canny(blurred, edges, 50, 150);
    // 결과 확인을 위해 에지 이미지 표시
    cv::imshow("Edges", edges);
    cv::waitKey(0);

    // 3. Hough 공간(파라미터 공간) 정의
    const int width = edges.cols;
    const int height = edges.rows;

    // rho(ρ) 최대값: 이미지 대각선 길이
    int max_rho = static_cast<int>(std::sqrt(width * width + height * height));
    int num_rho = 2 * max_rho; // -max_rho ~ +max_rho 범위
    int num_theta = 180;       // θ: 0 ~ 179도 (1도 간격)

    // 누산기(accumulator) 배열 생성 및 0으로 초기화
    std::vector<std::vector<int>> accumulator(num_rho, std::vector<int>(num_theta, 0));

    // θ 값 미리 계산 (라디안 단위)
    std::vector<double> thetas(num_theta);
    for (int t = 0; t < num_theta; ++t) {
        thetas[t] = t * CV_PI / num_theta;
    }

    // 코사인/사인 미리 계산
    std::vector<double> cos_t(num_theta), sin_t(num_theta);
    for (int t = 0; t < num_theta; ++t) {
        cos_t[t] = std::cos(thetas[t]);
        sin_t[t] = std::sin(thetas[t]);
    }

    // 4. 누산기 채우기: 에지 픽셀마다 rho 계산
    for (int y = 0; y < height; ++y) {
        const uchar* row = edges.ptr<uchar>(y);
        for (int x = 0; x < width; ++x) {
            if (row[x] == 0) continue; // 에지가 아닌 픽셀 스킵
            for (int t = 0; t < num_theta; ++t) {
                double rho = x * cos_t[t] + y * sin_t[t];
                int r = cvRound(rho) + max_rho; // 인덱스로 변환
                if (r >= 0 && r < num_rho) {
                    accumulator[r][t]++;
                }
            }
        }
    }

    // 5. 최고값 탐색 및 임계값 결정
    int max_votes = 0;
    for (int r = 0; r < num_rho; ++r)
        for (int t = 0; t < num_theta; ++t)
            if (accumulator[r][t] > max_votes) max_votes = accumulator[r][t];

    // 임계값: 데이터에 따라 다르게 설정 (최댓값의 비율 또는 절대값)
    int threshold = std::max(50, static_cast<int>(max_votes * 0.5));

    // 6. 피크 탐지 (간단한 비최대 억제)
    std::vector<std::pair<int,int>> peaks; // (r, t)
    int neighborhood_r = 10; // rho 이웃 범위
    int neighborhood_t = 10; // theta 이웃 범위

    for (int r = 0; r < num_rho; ++r) {
        for (int t = 0; t < num_theta; ++t) {
            int votes = accumulator[r][t];
            if (votes < threshold) continue;

            bool is_max = true;
            int r0 = std::max(0, r - neighborhood_r);
            int r1 = std::min(num_rho - 1, r + neighborhood_r);
            int t0 = std::max(0, t - neighborhood_t);
            int t1 = std::min(num_theta - 1, t + neighborhood_t);

            for (int rr = r0; rr <= r1 && is_max; ++rr) {
                for (int tt = t0; tt <= t1; ++tt) {
                    if (rr == r && tt == t) continue;
                    if (accumulator[rr][tt] > votes) { is_max = false; break; }
                }
            }

            if (is_max) peaks.emplace_back(r, t);
        }
    }

    std::cout << "Detected peaks: " << peaks.size() << " (threshold=" << threshold << ")" << std::endl;

    // 7. 검출된 피크를 이미지 상의 선분으로 변환 및 그리기
    cv::Mat result = src.clone();
    for (const auto &p : peaks) {
        int r = p.first;
        int t = p.second;
        double rho = r - max_rho;
        double theta = thetas[t];
        double a = std::cos(theta);
        double b = std::sin(theta);
        double x0 = a * rho;
        double y0 = b * rho;
        cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
        cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));
        cv::line(result, pt1, pt2, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
    }

    cv::imshow("Detected Lines", result);
    cv::imwrite("output.png", result);
    cv::waitKey(0);

    return 0;
}
