# 필요한 라이브러리를 설치해야 합니다.
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# 사전 훈련된 YOLOv8 모델을 로드합니다. 'yolov8n.pt'는 가장 작고 빠른 모델입니다.
# 성능이 더 좋은 모델을 원하시면 'yolov8s.pt', 'yolov8m.pt' 등을 사용할 수 있습니다.
model = YOLO('yolov8n.pt')

# 처리할 비디오 파일 경로를 지정합니다.
# 예: video_path = 'my_video.mp4'
# 웹캠을 사용하려면 video_path = 0 으로 설정하세요.
video_path = 'D:\\Downloads\\DrivingSample.mp4' # <<< 여기에 비디오 파일 경로를 입력하세요.
cap = cv2.VideoCapture(video_path)

# 비디오 캡처가 성공적으로 열렸는지 확인합니다.
if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {video_path}")
else:
    while cap.isOpened():
        # 비디오에서 한 프레임씩 읽어옵니다.
        success, frame = cap.read()

        if success:
            # YOLOv8 모델을 사용하여 프레임에서 객체를 탐지합니다.
            results = model(frame)
            
            # 탐지된 객체 정보를 프레임에 시각화합니다.
            annotated_frame = results[0].plot()

            # 결과 프레임을 화면에 표시합니다.
            cv2.imshow("YOLOv8 ADAS", annotated_frame)

            # 'q' 키를 누르면 루프를 종료합니다.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 비디오의 끝에 도달하면 루프를 종료합니다.
            print("비디오의 끝에 도달했습니다.")
            break

# 모든 리소스를 해제합니다.
cap.release()
cv2.destroyAllWindows()

print("프로그램을 종료합니다.")