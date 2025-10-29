// FeatureMatching.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp> // k-D 트리 사용을 위해 추가

int main(int argc, char** argv)
{
    // 얼굴 참조 이미지 경로: 실행 인수로 전달하거나 기본값 사용
    std::string ref_path = (argc > 1) ? argv[1] : "face.jpg";

    // 참조 이미지 로드 (컬러 -> 그레이 변환)
    cv::Mat ref_color = cv::imread(ref_path);
    if (ref_color.empty()) {
        std::cerr << "참조 이미지 로드 실패: " << ref_path << std::endl;
        std::cerr << "실행 예: FeatureMatching.exe face.jpg" << std::endl;
        return -1;
    }

    // 실습 중 크기 조절이 필요하면 여기서 변경 (너비 기준)
    const int ref_display_width = 300;
    if (ref_color.cols > ref_display_width) {
        double scale = double(ref_display_width) / ref_color.cols;
        cv::resize(ref_color, ref_color, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    cv::Mat ref_gray;
    cv::cvtColor(ref_color, ref_gray, cv::COLOR_BGR2GRAY);

    // SIFT 생성 (OpenCV가 SIFT를 지원해야 함)
    cv::Ptr<cv::SIFT> sift;
    try {
        sift = cv::SIFT::create();
    } catch (const cv::Exception& e) {
        std::cerr << "SIFT 생성 실패: " << e.what() << std::endl;
        std::cerr << "OpenCV가 SIFT를 지원하도록 빌드되어 있는지 확인하세요 (opencv_contrib 및 NONFREE 필요)." << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> kp_ref;
    cv::Mat desc_ref;
    sift->detectAndCompute(ref_gray, cv::noArray(), kp_ref, desc_ref);

    if (kp_ref.empty() || desc_ref.empty()) {
        std::cerr << "참조 이미지에서 특징을 찾을 수 없습니다." << std::endl;
        return -1;
    }

    // 웹캠 열기
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다." << std::endl;
        return -1;
    }

    // 참조 이미지의 코너 (원본 사이즈 기준)
    std::vector<cv::Point2f> ref_corners = {
        cv::Point2f(0, 0),
        cv::Point2f((float)ref_gray.cols, 0),
        cv::Point2f((float)ref_gray.cols, (float)ref_gray.rows),
        cv::Point2f(0, (float)ref_gray.rows)
    };

    cv::Mat frame, gray;
    std::cout << "실시간 매칭 시작. 종료: ESC 또는 'q'" << std::endl;

    const float ratio_thresh = 0.7f; // SIFT에 더 적합한 ratio
    const size_t min_good_matches = 10;

    while (true) {
        if (!cap.read(frame) || frame.empty()) break;

        // 화면 반전(웹캠이 미러처럼 보이길 원하면 주석 해제)
        // cv::flip(frame, frame, 1);

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // 프레임에서 키포인트/디스크립터 찾기
        std::vector<cv::KeyPoint> kp_frame;
        cv::Mat desc_frame;
        sift->detectAndCompute(gray, cv::noArray(), kp_frame, desc_frame);

        if (!desc_frame.empty()) {
            // --- 직접 구현한 특징 매칭 (k-D 트리 사용) ---
            // 1. k-D 트리를 사용하여 대상 영상(desc_frame)의 특징점으로 인덱스 생성
            cv::flann::Index flann_index(desc_frame, cv::flann::KDTreeIndexParams());

            // 2. 질의 영상(desc_ref)의 각 특징점에 대해 최근접 2개 이웃 탐색
            cv::Mat match_indices(desc_ref.rows, 2, CV_32S);
            cv::Mat match_dists(desc_ref.rows, 2, CV_32F);
            flann_index.knnSearch(desc_ref, match_indices, match_dists, 2, cv::flann::SearchParams());

            // 3. 최근접 1, 2위 거리 비율(Lowe's ratio test)을 이용해 좋은 매칭 선별
            std::vector<cv::DMatch> good_matches;
            for (int i = 0; i < desc_ref.rows; ++i) {
                if (match_dists.at<float>(i, 0) < ratio_thresh * match_dists.at<float>(i, 1)) {
                    // 좋은 매칭을 DMatch 객체로 만들어 저장
                    good_matches.emplace_back(i, match_indices.at<int>(i, 0), match_dists.at<float>(i, 0));
                }
            }
            // --- 매칭 구현 종료 ---

            // 충분한 좋은 매칭이 있으면 호모그래피 계산 (RANSAC 사용)
            if (good_matches.size() >= min_good_matches) {
                std::vector<cv::Point2f> pts_ref, pts_frame;
                pts_ref.reserve(good_matches.size());
                pts_frame.reserve(good_matches.size());
                for (const auto& gm : good_matches) {
                    pts_ref.push_back(kp_ref[gm.queryIdx].pt);
                    pts_frame.push_back(kp_frame[gm.trainIdx].pt);
                }

                cv::Mat mask;
                cv::Mat h_matrix = cv::findHomography(pts_ref, pts_frame, cv::RANSAC, 3.0, mask);

                if (!h_matrix.empty()) {
                    // 참조 이미지의 코너를 프레임 좌표로 투영
                    std::vector<cv::Point2f> projected_corners;
                    cv::perspectiveTransform(ref_corners, projected_corners, h_matrix);

                    // 테두리 그리기 (붉은색, 굵기 4)
                    std::vector<cv::Point> polygon;
                    for (auto &p : projected_corners) polygon.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
                    const cv::Scalar red(0, 0, 255);
                    const int thickness = 4;
                    cv::polylines(frame, polygon, true, red, thickness, cv::LINE_AA);

                    // 매칭 수 표시
                    std::string info = "Matches: " + std::to_string((int)good_matches.size());
                    cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, red, 2);
                }
            } else {
                // 매칭 부족 시 간단한 안내 텍스트
                std::string info = "Matches: " + std::to_string((int)good_matches.size());
                cv::putText(frame, info, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 200, 200), 2);
            }
        }

        // 왼쪽 상단에 참조 이미지 작게 보여주기 (디버깅/조정용)
        cv::Mat display;
        frame.copyTo(display);
        int w = ref_color.cols;
        int h = ref_color.rows;
        if (w > 0 && h > 0 && display.cols >= w && display.rows >= h) {
            cv::Rect roi(0, 0, w, h);
            ref_color.copyTo(display(roi));
            cv::rectangle(display, roi, cv::Scalar(255,255,255), 1);
        }

        cv::imshow("Feature Matching", display);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q') break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}

