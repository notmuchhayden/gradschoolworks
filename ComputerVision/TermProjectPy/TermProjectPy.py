# 필요한 라이브러리를 설치해야 합니다.
# pip install ultralytics opencv-python

import cv2
import socket
import json
from ultralytics import YOLO
from lane_utils import detect_lanes_hough

# 사전 훈련된 YOLOv8 모델을 로드합니다. 'yolov8n.pt'는 가장 작고 빠른 모델입니다.
# 성능이 더 좋은 모델을 원하시면 'yolov8s.pt', 'yolov8m.pt' 등을 사용할 수 있습니다.
model = YOLO('yolov8n.pt')

# 처리할 비디오 파일 경로를 지정합니다.
# 예: video_path = 'my_video.mp4'
# 웹캠을 사용하려면 video_path = 0 으로 설정하세요.
video_path = 'D:\\Downloads\\DrivingSample.mp4' # <<< 여기에 비디오 파일 경로를 입력하세요.
cap = cv2.VideoCapture(video_path)

# UDP 소켓 설정
UDP_IP = "127.0.0.1"  # Unity3D 서버 IP
UDP_PORT = 5005       # Unity3D 서버 포트
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 비디오 캡처가 성공적으로 열렸는지 확인합니다.
if not cap.isOpened():
    print(f"오류: 비디오 파일을 열 수 없습니다. 경로를 확인하세요: {video_path}")
else:
    paused = False
    running = True
    while cap.isOpened() and running:
        # 비디오에서 한 프레임씩 읽어옵니다.
        success, frame = cap.read()

        if success:
            # YOLOv8 모델을 사용하여 프레임에서 객체를 탐지합니다.
            results = model(frame)
            boxes = results[0].boxes
            data_to_send = []

            if boxes is not None and boxes.xyxy is not None:
                for i in range(len(boxes)):
                    box = boxes.xyxy[i].cpu().numpy().tolist()  # [x1, y1, x2, y2]
                    cls = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    data_to_send.append({
                        "class": cls,
                        "confidence": conf,
                        "bbox": box
                    })

            # JSON 문자열로 변환 후 UDP 전송
            message = json.dumps(data_to_send)
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

            # YOLO 어노테이트(원하면)와 차선 어노테이트를 합성
            yolo_annot = results[0].plot()
            lane_annot = detect_lanes_hough(frame)
            combined = cv2.addWeighted(yolo_annot, 0.7, lane_annot, 0.3, 0)
            cv2.imshow("YOLOv8 ADAS", combined)

            # 키 입력 처리: 'p'로 일시정지 토글, 'q'로 종료
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                paused = True

            # 일시정지 상태일 때: 'p'로 재개, 'q'로 종료
            while paused:
                k = cv2.waitKey(0) & 0xFF
                if k == ord('p'):
                    paused = False
                    break
                elif k == ord('q'):
                    paused = False
                    running = False
                    break
        else:
            # 비디오의 끝에 도달하면 루프를 종료합니다.
            print("비디오의 끝에 도달했습니다.")
            break

# 모든 리소스를 해제합니다.
cap.release()
cv2.destroyAllWindows()
sock.close()

print("프로그램을 종료합니다.")