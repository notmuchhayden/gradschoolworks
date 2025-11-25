# 필요한 라이브러리를 설치해야 합니다.
# pip install ultralytics opencv-python

import cv2
import socket
import json
from ultralytics import YOLO
from lane_utils import detect_lanes_hough, visualize_region_of_interest, ROIConfig

# 사전 훈련된 YOLOv8 모델을 로드합니다. 'yolov8n.pt'는 가장 작고 빠른 모델입니다.
# 성능이 더 좋은 모델을 원하시면 'yolov8s.pt', 'yolov8m.pt' 등을 사용할 수 있습니다.
model = YOLO('yolov8n.pt')

# 처리할 비디오 파일 경로를 지정합니다.
# 예: video_path = 'my_video.mp4'
# 웹캠을 사용하려면 video_path =0 으로 설정하세요.
video_path = 'D:\\Downloads\\DrivingSample.mp4' # <<< 여기에 비디오 파일 경로를 입력하세요.
cap = cv2.VideoCapture(video_path)

# UDP 소켓 설정
UDP_IP = "127.0.0.1" # Unity3D 서버 IP
UDP_PORT =5005 # Unity3D 서버 포트
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# ROI 설정 (공유 구조체)
roi = ROIConfig()
show_roi_overlay = True

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
                    box = boxes.xyxy[i].cpu().numpy().tolist() # [x1, y1, x2, y2]
                    cls = int(boxes.cls[i].cpu().numpy())
                    conf = float(boxes.conf[i].cpu().numpy())
                    data_to_send.append({
                        "type": "cars",
                        "class": cls,
                        "confidence": conf,
                        "bbox": box,
                    })

            # YOLO 어노테이트(원하면)와 차선 어노테이트를 합성
            yolo_annot = results[0].plot()
            lane_annot, left_lane, right_lane = detect_lanes_hough(frame, roi_config=roi)

            # 차선 정보를 data_to_send에 추가
            lane_data = {}
            if left_lane is not None:
                lane_data["left_lane"] = left_lane
            if right_lane is not None:
                lane_data["right_lane"] = right_lane
            
            if lane_data:
                data_to_send.append({
                    "type": "lanes",
                    "data": lane_data
                })

            # JSON 문자열로 변환 후 UDP 전송
            message = json.dumps(data_to_send)
            sock.sendto(message.encode(), (UDP_IP, UDP_PORT))

            # ROI 시각화를 원하면 combined 이미지 위에 visualize_region_of_interest를 적용
            combined = cv2.addWeighted(yolo_annot,0.7, lane_annot,0.3,0)
            if show_roi_overlay:
                combined_with_roi = visualize_region_of_interest(combined, color=(255,0,0), alpha=0.25, roi_config=roi)
            else:
                combined_with_roi = combined

            # 화면에 현재 ROI 파라미터 출력
            info_text = (
                f"ROI: bl={roi.bottom_left:.2f} tl={roi.top_left:.2f} tr={roi.top_right:.2f} "
                f"br={roi.bottom_right:.2f} ty={roi.top_y:.2f} | r:toggle ROI i/k:top_y +/- j/l:shrink/expand top"
            )
            cv2.putText(combined_with_roi, info_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,255),1, cv2.LINE_AA)

            cv2.imshow("YOLOv8 ADAS", combined_with_roi)

            # 키 입력 처리: 'p'로 일시정지 토글, 'q'로 종료
            key = cv2.waitKey(1) &0xFF
            if key == ord('q'):
                break
            if key == ord('p'):
                paused = True

            # ROI 토글 및 조정 키
            if key == ord('r'):
                show_roi_overlay = not show_roi_overlay
            if key == ord('i'):
                roi.top_y = min(0.9, roi.top_y +0.01)
            if key == ord('k'):
                roi.top_y = max(0.2, roi.top_y -0.01)
            # j: shrink top width, l: expand top width
            if key == ord('j'):
                # shrink: move top edges toward center
                roi.top_left = min(roi.top_left +0.01, roi.top_right -0.01)
                roi.top_right = max(roi.top_right -0.01, roi.top_left +0.01)
            if key == ord('l'):
                # expand: move top edges outward
                roi.top_left = max(0.0, roi.top_left -0.01)
                roi.top_right = min(1.0, roi.top_right +0.01)

        else:
            # 비디오의 끝에 도달하면 루프를 종료합니다.
            print("비디오의 끝에 도달했습니다.")
            break

        # 일시정지 상태일 때: 'p'로 재개, 'q'로 종료
        while paused:
            k = cv2.waitKey(0) &0xFF
            if k == ord('p'):
                paused = False
                break
            elif k == ord('q'):
                paused = False
                running = False
                break
            elif k == ord('r'):
                show_roi_overlay = not show_roi_overlay
            elif k == ord('i'):
                roi.top_y = min(0.9, roi.top_y +0.01)
            elif k == ord('k'):
                roi.top_y = max(0.2, roi.top_y -0.01)
            elif k == ord('j'):
                roi.top_left = min(roi.top_left +0.01, roi.top_right -0.01)
                roi.top_right = max(roi.top_right -0.01, roi.top_left +0.01)
            elif k == ord('l'):
                roi.top_left = max(0.0, roi.top_left -0.01)
                roi.top_right = min(1.0, roi.top_right +0.01)

# 모든 리소스를 해제합니다.
cap.release()
cv2.destroyAllWindows()
sock.close()

print("프로그램을 종료합니다.")