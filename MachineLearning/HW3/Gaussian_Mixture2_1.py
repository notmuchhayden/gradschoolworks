import os
import numpy as np
import h5py # Changed from scipy.io import loadmat
import matplotlib.pyplot as plt

# 문제 2-(1)
# 가우시안 혼합 모델을 사용하여 주어진 데이터 HW3Data.mat에 대한 분석을 수행 하시오.
# 데이터 HW3Data.mat을 불러온 다음 2차원 평면에 산점도를 그리시오.
# (scatter 함수 사용할 것)

# 현재 파일 경로 기준으로 .mat 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(script_dir, 'HW3Data.mat')

# 데이터 로드 (h5py 사용으로 변경)
with h5py.File(mat_file_path, 'r') as f:
    # .mat 파일 (v7.3+) 내부의 데이터셋 이름이 'data'라고 가정합니다.
    # h5py로 직접 읽을 때, scipy.io.loadmat과 배열 방향이 다를 수 있어 .T (전치)를 사용합니다.
    # 만약 데이터가 이미 (샘플 수, 특징 수) 형태로 저장되어 있다면 .T는 필요 없을 수 있습니다.
    X = np.array(f['data']).T.astype(float)

# 데이터 시각화
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], alpha=0.6) # X의 첫 번째 열을 x축, 두 번째 열을 y축으로 사용
plt.title('Scatter Plot of HW3Data.mat (loaded with h5py)') # 제목 수정
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.grid(True)
plt.show()
