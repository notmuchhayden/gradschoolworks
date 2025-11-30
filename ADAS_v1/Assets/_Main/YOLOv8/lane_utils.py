from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class ROIConfig:
    bottom_left: float = 0.01
    top_left: float = 0.42
    top_right: float = 0.58
    bottom_right: float = 0.99
    top_y: float = 0.55

def color_mask_hls(img):
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 흰색 마스크
    lower_white = np.array([0, 200, 0])
    upper_white = np.array([180, 255, 255])
    mask_white = cv2.inRange(hls, lower_white, upper_white)
    # 노란색 마스크
    lower_yellow = np.array([15, 30, 115])
    upper_yellow = np.array([35, 204, 255])
    mask_yellow = cv2.inRange(hls, lower_yellow, upper_yellow)
    return cv2.bitwise_or(mask_white, mask_yellow)

def _polygon_from_roi(img_shape, roi_config: ROIConfig):
    h, w = img_shape[:2]
    polygon = np.array([
        (int(roi_config.bottom_left * w), h),
        (int(roi_config.top_left * w), int(roi_config.top_y * h)),
        (int(roi_config.top_right * w), int(roi_config.top_y * h)),
        (int(roi_config.bottom_right * w), h),
    ], np.int32)
    return polygon

def region_of_interest(img, roi_config: ROIConfig = None):
    
    if roi_config is None:
        roi_config = ROIConfig()

    mask = np.zeros_like(img)
    polygon = _polygon_from_roi(img.shape, roi_config)
    polygon_fill = np.array([polygon], np.int32)

    if len(img.shape) > 2:
        cv2.fillPoly(mask, polygon_fill, (255,) * img.shape[2])
    else:
        cv2.fillPoly(mask, polygon_fill, 255)
    return cv2.bitwise_and(img, mask)

def visualize_region_of_interest(img, color=(0,255,0), alpha=0.3, roi_config: ROIConfig = None):
    
    if roi_config is None:
        roi_config = ROIConfig()

    # 그레이 이미지면 컬러로 변환
    if len(img.shape) == 2 or img.shape[2] == 1:
        vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        vis = img.copy()

    polygon = _polygon_from_roi(vis.shape, roi_config)

    overlay = vis.copy()
    cv2.fillPoly(overlay, [polygon], color)
    # 반투명 합성
    output = cv2.addWeighted(overlay, alpha, vis, 1 - alpha, 0)
    # ROI 경계선도 그려 가시성 향상
    cv2.polylines(output, [polygon], isClosed=True, color=(0, 0, 0), thickness=2)
    return output

def average_slope_intercept(lines, img_shape):
    left_lines = []
    right_lines = []
    if lines is None:
        return None, None
    
    h, w = img_shape[:2]
    center_x = w / 2  # 화면 중심 x 좌표
    
    # 얻어진 직선에서 기울기와 절편 계산
    # 선분의 중심점이 화면 중심 기준 좌우 어디에 있는지로 분류
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4:
            continue
        intercept = y1 - slope * x1
        
        # 선분의 중심 x 좌표 계산
        mid_x = (x1 + x2) / 2
        
        # 화면 중심을 기준으로 좌우 분류
        if mid_x < center_x:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    
    h = img_shape[0] # 이미지 높이
    
    def make_line(lines_list):
        if not lines_list:
            return None
        # RANSAC 또는 중앙값 사용 (이상치 제거)
        slopes, intercepts = zip(*lines_list)
        # 기울기와 절편의 중앙값을 계산하여 대표값 획득
        slope = np.median(slopes)
        intercept = np.median(intercepts)
        
        # 이미지 하단을 기준으로 60% 지점까지 y 좌표 설정
        y1 = h
        y2 = int(h * 0.6)

        # slope가 0에 가까우면 None 반환
        if abs(slope) < 0.01:
            return None

        # 기울기와 절편으로부터 x 좌표 계산
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)

        return np.array([x1, y1, x2, y2])
    
    left = make_line(left_lines)
    right = make_line(right_lines)
    return left, right

def detect_lanes_hough(frame, roi_config: ROIConfig = None):
    # 입력: BGR 이미지, 출력: (BGR 이미지(차선이 그려진), 좌우 차선 정보)
    if roi_config is None:
        roi_config = ROIConfig()

    img = frame.copy()
    # 1. 색상 마스크
    mask = color_mask_hls(img)
    masked = cv2.bitwise_and(img, img, mask=mask)
    # 2. 그레이 + 블러 + 캐니
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # 3. ROI
    edges_roi = region_of_interest(edges, roi_config=roi_config)
    # 4. HoughLinesP
    lines = cv2.HoughLinesP(edges_roi, rho=1, theta=np.pi / 180, threshold=50, minLineLength=30, maxLineGap=50)
    line_img = np.zeros_like(img)

    # 평균 기울기-절편 방법으로 좌우 차선 계산
    left, right = average_slope_intercept(lines, img.shape)
    
    # 차선 그리기
    if left is not None:
        cv2.line(line_img, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 8)
    if right is not None:
        cv2.line(line_img, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 8)

    # 5. 합성
    combo = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    
    # 차선 정보를 딕셔너리로 반환 (None이면 빈 딕셔너리)
    left_lane = {"x1": int(left[0]), "y1": int(left[1]), "x2": int(left[2]), "y2": int(left[3])} if left is not None else None
    right_lane = {"x1": int(right[0]), "y1": int(right[1]), "x2": int(right[2]), "y2": int(right[3])} if right is not None else None
    
    return combo, left_lane, right_lane