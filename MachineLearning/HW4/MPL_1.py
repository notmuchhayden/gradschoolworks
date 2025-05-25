# 1. (100점) 참고교재의 프로그램 11-1을 참고하여, 주어진 데이터 HW4data을 이용하여 다층
# 퍼셉트론을 구현하시오. 데이터의 변수 X는 400개의 2차원 샘플로 구성되었으며, T는 400
# 개 데이터에 대한 클래스 레이블로 1과 –1로 구분되어 있음.
# (단, 종료조건으로 최대 반복횟수는 1,000회 그리고 학습오차 0.05 미만으로 설정하시오)

# (1) (10점) 데이터 HW4data을 불러온 다음 클래스를 구분되도록 색/모양을 달리하여 2차원
# 평면에 산점도를 그리시오.

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(script_dir, 'HW4data_Python.mat')
data = sio.loadmat(mat_file_path)

X = data['X'] # 2차원 샘플 데이터 (400, 2)
T = data['T'] # 1차원 클래스 레이블 (400, 1)

# 데이터 시각화
# 클래스 레이블이 1인 데이터와 -1인 데이터를 분리합니다.
X_class1 = X[T[:, 0] == 1]
X_class_minus1 = X[T[:, 0] == -1]

plt.figure(figsize=(8, 6))

# 클래스 1 데이터를 파란색 'o' 마커로 플롯합니다.
plt.scatter(X_class1[:, 0], X_class1[:, 1], c='blue', marker='o', label='Class 1')

# 클래스 -1 데이터를 빨간색 'x' 마커로 플롯합니다.
plt.scatter(X_class_minus1[:, 0], X_class_minus1[:, 1], c='red', marker='x', label='Class -1')

plt.title('Scatter Plot of HW4data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.show()