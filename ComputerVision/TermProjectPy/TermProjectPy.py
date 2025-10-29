# �ʿ��� ���̺귯���� ��ġ�ؾ� �մϴ�.
# pip install ultralytics opencv-python

import cv2
from ultralytics import YOLO

# ���� �Ʒõ� YOLOv8 ���� �ε��մϴ�. 'yolov8n.pt'�� ���� �۰� ���� ���Դϴ�.
# ������ �� ���� ���� ���Ͻø� 'yolov8s.pt', 'yolov8m.pt' ���� ����� �� �ֽ��ϴ�.
model = YOLO('yolov8n.pt')

# ó���� ���� ���� ��θ� �����մϴ�.
# ��: video_path = 'my_video.mp4'
# ��ķ�� ����Ϸ��� video_path = 0 ���� �����ϼ���.
video_path = 'path_to_your_video.mp4' # <<< ���⿡ ���� ���� ��θ� �Է��ϼ���.
cap = cv2.VideoCapture(video_path)

# ���� ĸó�� ���������� ���ȴ��� Ȯ���մϴ�.
if not cap.isOpened():
    print(f"����: ���� ������ �� �� �����ϴ�. ��θ� Ȯ���ϼ���: {video_path}")
else:
    while cap.isOpened():
        # �������� �� �����Ӿ� �о�ɴϴ�.
        success, frame = cap.read()

        if success:
            # YOLOv8 ���� ����Ͽ� �����ӿ��� ��ü�� Ž���մϴ�.
            results = model(frame)

            # Ž���� ��ü ������ �����ӿ� �ð�ȭ�մϴ�.
            annotated_frame = results[0].plot()

            # ��� �������� ȭ�鿡 ǥ���մϴ�.
            cv2.imshow("YOLOv8 ADAS", annotated_frame)

            # 'q' Ű�� ������ ������ �����մϴ�.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # ������ ���� �����ϸ� ������ �����մϴ�.
            print("������ ���� �����߽��ϴ�.")
            break

# ��� ���ҽ��� �����մϴ�.
cap.release()
cv2.destroyAllWindows()

print("���α׷��� �����մϴ�.")