import os
import numpy as np
from scipy.io import loadmat

# 문제 1-(1)
# 가우시안 혼합 모델을 사용하여 주어진 데이터 HW3Data.mat에 대한 분석을 수행 하시오.
# 데이터 HW3Data.mat을 불러온 다음 2차원 평면에 산점도를 그리시오.
# (scatter 함수 사용할 것)

# 현재 파일 경로 기준으로 .mat 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(script_dir, 'HW3Data.mat')

# 데이터 로드
iris_data = loadmat(mat_file_path)
X = iris_data['iris_data'].astype(float)

