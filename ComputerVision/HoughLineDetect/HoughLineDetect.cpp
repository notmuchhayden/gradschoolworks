// HoughLineDetect.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
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
	// 결과 확인을 위해	에지 이미지 표시
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

	return 0;
}
