
/*
세 번째 과제입니다.

금번 과제는 지난 과제와 마찬가지로, 다음의 내용을 직접 구현해보는 것입니다.
- 9주차 학습에서 배운 내용을 토대로 "특징의 매칭"을 실습 해보는 과제입니다.

1. 자신의 얼굴 사진을 적당한 크기로(실습을 진행하면서 크기 조정 필요) 준비해서 찾고자 하는 특징 이미지로 합니다.
2. 웹캠을 통해서 실시간으로 입력 받은 여러분의 영상에서(거리 조절 필요) 얼굴을 매칭시키는 코드를 작성하는 것입니다.
3. 얼굴 사진과 매칭되는 영역(사각형 또는 원)에는 빨간색 굵은 테두리로 표시를 합니다.

- 웹캠이 없으신 분들은 동영상 자료를(스마트폰으로 녹화 가능) 활용해도 됩니다.
- 스마트폰을 웹캠 대신으로 사용가능합니다(사용법은 인터넷검색)

이번 과제의 제출 기한은 2주를 드립니다.

Visual Studio 도구를 사용하여 C/C++ 언어를 사용하여 구현한 경우에는 솔루션/프로젝트 전체를 압축하여 제출해주시면 됩니다.
※ 단, 디버깅 및 .vs(숨겨짐) 폴더는 삭제하고 압축해주세요~ / 그렇지 않으면 용량 초과로 제출 안될수도 있습니다.
※ 제출 파일 이름 : 이름_과제3.zip
*/

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#define MINIMUM_GOOD_MATCHES 10
#define MATCH_THRESHOLD 0.75f

int main()
{
    // 얼굴 참조 이미지 경로
    std::string ref_path = "../assets/face_crop.jpg";

    // 참조 이미지 로드
    cv::Mat query_img_color = cv::imread(ref_path);
    if (query_img_color.empty()) {
        std::cerr << "얼굴 이미지 로드 실패: " << ref_path << std::endl;
        return -1;
    }

	// 그레이스케일 변환
    cv::Mat query_img_gray;
    cv::cvtColor(query_img_color, query_img_gray, cv::COLOR_BGR2GRAY);

    // SIFT 로 특징 생성
    cv::Ptr<cv::SIFT> sift;
    try {
        sift = cv::SIFT::create();
    } catch (const cv::Exception& e) {
        // OpenCV가 SIFT를 지원하도록 빌드되어 있는지 확인하세요 (opencv_contrib 및 NONFREE 필요).
        std::cerr << "SIFT 생성 실패: " << e.what() << std::endl;
        return -1;
    }

    std::vector<cv::KeyPoint> keypoint_face; // 얼굴 이미지 키포인트
	cv::Mat desc_face; // 얼굴 이미지 디스크립터
    sift->detectAndCompute(query_img_gray, cv::noArray(), keypoint_face, desc_face);

    if (keypoint_face.empty() || desc_face.empty()) {
        std::cerr << "얼굴 이미지에서 특징을 찾을 수 없습니다." << std::endl;
        return -1;
    }

    // 얼굴 원본 이미지 위에 키포인트를 그림
    cv::Mat query_img_kp_display;
    cv::drawKeypoints(query_img_color, keypoint_face, query_img_kp_display,
                      cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	// Thumbnail 크기로 조정
    const int query_display_width = 100;
    if (query_img_color.cols > query_display_width) {
        double scale = double(query_display_width) / query_img_color.cols;
        cv::resize(query_img_color, query_img_color, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // 창에 표시 혹은 파일로 저장
    cv::imshow("Keypoints", query_img_kp_display);
    //cv::imwrite("../assets/face_keypoints.jpg", query_img_kp_display);
    cv::waitKey(0);

    // 웹캠 열기
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "웹캠을 열 수 없습니다." << std::endl;
        return -1;
    }

	// 참조 이미지 RECT 좌표, 호모그래피 변환용
    std::vector<cv::Point2f> recPts = {
        cv::Point2f(0, 0),
        cv::Point2f((float)query_img_gray.cols, 0),
        cv::Point2f((float)query_img_gray.cols, (float)query_img_gray.rows),
        cv::Point2f(0, (float)query_img_gray.rows)
    };

    cv::Mat cam_frame_color, cam_frame_gray;
    std::cout << "실시간 매칭 시작. 종료: ESC 또는 'q'" << std::endl;

	// Brute-Force 매처 생성 (L1 거리 사용)
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create(cv::NORM_L1);
    while (true) {
        // 캠에서 한프레임 읽기
        if (!cap.read(cam_frame_color) || cam_frame_color.empty())
            break;

        // 화면 반전(웹캠이 미러처럼 보이길 원하면 주석 해제)
        // cv::flip(cam_frame_color, cam_frame_color, 1);
		// 캠 화면을 그레이스케일 변환
        cv::cvtColor(cam_frame_color, cam_frame_gray, cv::COLOR_BGR2GRAY);

        // 이번 프레임에서 키포인트, 디스크립터 찾기
        std::vector<cv::KeyPoint> keypoint_frame;
        cv::Mat desc_frame;
        sift->detectAndCompute(cam_frame_gray, cv::noArray(), keypoint_frame, desc_frame);
        
        if (!desc_frame.empty()) {
            // knnMatch로 2개 이웃 검색
            std::vector<std::vector<cv::DMatch>> matches;
			matcher->knnMatch(desc_face, desc_frame, matches, 2);
            
            // Lowe's ratio test 적용하여 좋은 매칭 선별
            std::vector<cv::DMatch> good_matches;
            good_matches.reserve(matches.size());
            for (const auto& m : matches) {
                if (m.size() >= 2) {
                    const cv::DMatch& f = m[0]; // 1등 매칭
					const cv::DMatch& s = m[1]; // 2등 매칭
                    if (f.distance < s.distance * MATCH_THRESHOLD) {
                        good_matches.push_back(f); // f: query->train
                    }
                }
            }

            // 충분한 좋은 매칭이 있으면 호모그래피 계산 (RANSAC 사용)
            if (good_matches.size() >= MINIMUM_GOOD_MATCHES) {
                std::vector<cv::Point2f> pts_face, pts_frame;
                pts_face.reserve(good_matches.size());
                pts_frame.reserve(good_matches.size());
                for (const auto& gm : good_matches) {
                    pts_face.push_back(keypoint_face[gm.queryIdx].pt);
                    pts_frame.push_back(keypoint_frame[gm.trainIdx].pt);
                }

				// RANSAC으로 호모그래피 행렬 계산
                cv::Mat mask;
                cv::Mat mtrx = cv::findHomography(pts_face, pts_frame, cv::RANSAC, 3.0, mask);

                if (!mtrx.empty()) {
                    // 참조 이미지의 코너를 프레임 좌표로 투영
                    std::vector<cv::Point2f> dst;
                    cv::perspectiveTransform(recPts, dst, mtrx);

                    // 호모그래피 테두리 그리기
                    std::vector<cv::Point> polygon;
                    for (auto& p : dst) {
                        polygon.emplace_back(cv::Point(cvRound(p.x), cvRound(p.y)));
                    }
                    const cv::Scalar red(0, 0, 255);
                    const int thickness = 2;
                    cv::polylines(cam_frame_color, polygon, true, red, thickness, cv::LINE_AA);

                    // 매칭 수 표시
                    std::string info = "Matches: " + std::to_string((int)good_matches.size());
                    cv::putText(cam_frame_color, info, cv::Point(110, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, red, 2);
                }
            } 
        }

        // 왼쪽 상단에 참조 이미지 작게 보여주기
        cv::Mat display;
        cam_frame_color.copyTo(display);
        int w = query_img_color.cols;
        int h = query_img_color.rows;
        if (w > 0 && h > 0 && display.cols >= w && display.rows >= h) {
            cv::Rect roi(0, 0, w, h);
            query_img_color.copyTo(display(roi));
            cv::rectangle(display, roi, cv::Scalar(255,255,255), 1);
        }

        cv::imshow("Feature Matching", display);

        char key = (char)cv::waitKey(1);
        if (key == 27 || key == 'q' || key == 'Q')
            break;
    }

    // 메모리 해제
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
