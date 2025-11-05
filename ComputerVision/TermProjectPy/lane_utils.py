import cv2
import numpy as np

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

def region_of_interest(img):
    h, w = img.shape[:2]
    mask = np.zeros_like(img)
    # 아래 삼각형 ROI (비율로 지정)
    polygon = np.array([[
        (int(0.1*w), h),
        (int(0.45*w), int(0.6*h)),
        (int(0.55*w), int(0.6*h)),
        (int(0.9*w), h)
    ]], np.int32)
    if len(img.shape) > 2:
        cv2.fillPoly(mask, polygon, (255,)*img.shape[2])
    else:
        cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)

def average_slope_intercept(lines, img_shape):
    left_lines = []
    right_lines = []
    if lines is None:
        return None, None
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        if x2==x1:
            continue
        slope = (y2-y1)/(x2-x1)
        if abs(slope) < 0.4: # 거의 수평선 무시
            continue
        intercept = y1 - slope*x1
        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))
    h = img_shape[0]
    def make_line(avg):
        slope, intercept = avg
        y1 = h
        y2 = int(h*0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1,y1,x2,y2])
    left_avg = np.mean(left_lines, axis=0) if left_lines else None
    right_avg = np.mean(right_lines, axis=0) if right_lines else None
    left = make_line(left_avg) if left_avg is not None else None
    right = make_line(right_avg) if right_avg is not None else None
    return left, right

def detect_lanes_hough(frame):
    # 입력: BGR 이미지, 출력: BGR 이미지(차선이 그려진)
    img = frame.copy()
    # 1. 색상 마스크
    mask = color_mask_hls(img)
    masked = cv2.bitwise_and(img, img, mask=mask)
    # 2. 그레이 + 블러 + 캐니
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)
    # 3. ROI
    edges_roi = region_of_interest(edges)
    # 4. HoughLinesP
    lines = cv2.HoughLinesP(edges_roi, rho=1, theta=np.pi/180, threshold=50, minLineLength=30, maxLineGap=50)
    left, right = average_slope_intercept(lines, img.shape)
    line_img = np.zeros_like(img)
    if left is not None:
        cv2.line(line_img, (left[0], left[1]), (left[2], left[3]), (0,255,0), 8)
    if right is not None:
        cv2.line(line_img, (right[0], right[1]), (right[2], right[3]), (0,255,0), 8)
    # 5. 합성
    combo = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return combo