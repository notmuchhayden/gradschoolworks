import os
import numpy as np
from scipy.io import loadmat

# 현재 파일 경로 기준으로 .mat 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(script_dir, 'iris_shuffled.mat')

# 데이터 로드
iris_data = loadmat(mat_file_path)
X = iris_data['iris_data'].astype(float)

# 각 클래스별 데이터 분할 인덱스 정의
class_size = 50
train_size_per_class = 40
test_size_per_class = 10

train_indices = []
test_indices = []

for i in range(3): # 3개의 클래스에 대해 반복
    start_index = i * class_size
    # 훈련 데이터 인덱스: 각 클래스의 처음 40개
    train_indices.extend(range(start_index, start_index + train_size_per_class))
    # 테스트 데이터 인덱스: 각 클래스의 나머지 10개
    test_indices.extend(range(start_index + train_size_per_class, start_index + class_size))

# 인덱스를 사용하여 데이터 분할
train_data = X[train_indices, :]
test_data = X[test_indices, :]


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
            #if idx < 50:
            if idx < data_size/3:
                c[0] += 1
            #elif idx >= 100:
            elif idx >= data_size/3 * 2:
                c[2] += 1
            #elif 50 <= idx < 100:
            elif data_size/3 <= idx < data_size/3 * 2:
                c[1] += 1

        maxi = np.argmax(c)
        #true_label = (i // 50)
        true_label = (i // data_size/3)

        if maxi != true_label:
            Etrain += 1
    return Etrain

# 테스트할 K 값들
k_values = [5, 10, 15, 20, 25, 30]
results = [] # 결과를 저장할 리스트 (K, 오분류 수, 오분류율)

N = train_data.shape[0]
# 각 K 값에 대해 오분류 수 및 오분류율 계산
print("Calculating errors for different K values...")
for k in k_values:
    error_count = classify_KNN(k, N, train_data)
    error_rate = error_count / N
    results.append((k, error_count, error_rate))

# 결과를 텍스트 표로 출력
print("\n--- KNN Error Rate Summary for train data ---")
print("----------------------------")
# 헤더 출력 (열 너비와 정렬 지정)
print(f"{'K':<5} | {'Error Count':<12} | {'Error Rate':<10}")
print("----------------------------")
# 데이터 행 출력 (형식 지정)
for k, count, rate in results:
    # f-string을 사용하여 각 열의 너비와 정렬, 소수점 자릿수 지정
    print(f"{k:<5} | {count:<12} | {rate:<10.4f}")
print("----------------------------")

results = []
N = test_data.shape[0]
# 각 K 값에 대해 오분류 수 및 오분류율 계산
print("Calculating errors for different K values...")
for k in k_values:
    error_count = classify_KNN(k, N, test_data)
    error_rate = error_count / N
    results.append((k, error_count, error_rate))

# 결과를 텍스트 표로 출력
print("\n--- KNN Error Rate Summary for test data ---")
print("----------------------------")
# 헤더 출력 (열 너비와 정렬 지정)
print(f"{'K':<5} | {'Error Count':<12} | {'Error Rate':<10}")
print("----------------------------")
# 데이터 행 출력 (형식 지정)
for k, count, rate in results:
    # f-string을 사용하여 각 열의 너비와 정렬, 소수점 자릿수 지정
    print(f"{k:<5} | {count:<12} | {rate:<10.4f}")
print("----------------------------")
