from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class ROIConfig:
    bottom_left: float = 0.01
    top_left: float = 0.45
    top_right: float = 0.55
    bottom_right: float = 0.99
    top_y: float = 0.6

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
    """
    이미지에서 관심영역(ROI)을 마스킹합니다. roi_config을 전달하면 해당 비율로 폴리곤을 계산합니다.
    """
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
    """
    입력 이미지 위에 ROI 폴리곤을 반투명으로 그려서 시각화합니다.
    roi_config을 전달하면 해당 폴리곤을 사용합니다.
    반환: ROI가 오버레이된 BGR 이미지
    """
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
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.4: # 거의 수평선 무시
            continue
        intercept = y1 - slope * x1
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    h = img_shape[0]
    def make_line(avg):
        slope, intercept = avg
        y1 = h
        y2 = int(h * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None
    left = make_line(left_avg) if left_avg is not None else None
    right = make_line(right_avg) if right_avg is not None else None
    return left, right

def detect_lanes_hough(frame, roi_config: ROIConfig = None):
    # 입력: BGR 이미지, 출력: BGR 이미지(차선이 그려진)
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
    left, right = average_slope_intercept(lines, img.shape)
    line_img = np.zeros_like(img)
    if left is not None:
        cv2.line(line_img, (left[0], left[1]), (left[2], left[3]), (0, 255, 0), 8)
    if right is not None:
        cv2.line(line_img, (right[0], right[1]), (right[2], right[3]), (0, 255, 0), 8)
    # 5. 합성
    combo = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return combo