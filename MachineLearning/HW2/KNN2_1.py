import os
import numpy as np
from scipy.io import loadmat

# 윈도환경에서 같은 디렉토리내에 *.py 와 *.mat 이 있어도 파일을 읽지 못하는 현상이 있어서 추가
# 스크립트 파일이 있는 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))
# .mat 파일의 절대 경로 생성
mat_file_path = os.path.join(script_dir, 'iris_shuffled.mat')

# 데이터 로드
iris_data = loadmat(mat_file_path)
X = iris_data['iris_data'].astype(float)  # iris_data 변수 사용

N = X.shape[0]

def classify_KNN(K, data_size, data):
    Etrain = 0
    for i in range(data_size):
        x = data[i, :]  # 각 데이터에 대해 분류 시작
        d = np.zeros((data_size, 1))
        
        # 모든 데이터와의 거리 계산
        for j in range(data_size):
            d[j, 0] = np.linalg.norm(x - data[j, :])

        # 거리순으로 정렬
        si = np.argsort(d[:, 0])        
        c = np.zeros(3)

        # 이웃한 K개 데이터의 라벨을 점검하여 투표 수행
        for j in range(K):
            idx = si[j]
            if idx < 50:
                c[0] += 1
            elif idx >= 100:
                c[2] += 1
            elif 50 <= idx < 100:
                c[1] += 1

        maxi = np.argmax(c)
        true_label = (i // 50)

        if maxi != true_label:
            Etrain += 1
    return Etrain

# 테스트할 K 값들
k_values = [5, 10, 15, 20, 25, 30]
results = [] # 결과를 저장할 리스트 (K, 오분류 수, 오분류율)

# 각 K 값에 대해 오분류 수 및 오분류율 계산
print("Calculating errors for different K values...")
for k in k_values:
    error_count = classify_KNN(k, N, X)
    error_rate = error_count / N
    results.append((k, error_count, error_rate))

# 결과를 텍스트 표로 출력
print("\n--- KNN Error Rate Summary ---")
print("----------------------------")
# 헤더 출력 (열 너비와 정렬 지정)
print(f"{'K':<5} | {'Error Count':<12} | {'Error Rate':<10}")
print("----------------------------")
# 데이터 행 출력 (형식 지정)
for k, count, rate in results:
    # f-string을 사용하여 각 열의 너비와 정렬, 소수점 자릿수 지정
    print(f"{k:<5} | {count:<12} | {rate:<10.4f}")
print("----------------------------")
