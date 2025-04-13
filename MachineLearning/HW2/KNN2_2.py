import os
import numpy as np
from scipy.io import loadmat

# 윈도환경에서 같은 디렉토리내에 *.py 와 *.mat 이 있어도 파일을 읽지 못하는 현상이 있어서 추가
# 스크립트 파일이 있는 디렉토리 경로 가져오기
script_dir = os.path.dirname(os.path.abspath(__file__))
# .mat 파일의 절대 경로 생성
mat_file_path = os.path.join(script_dir, 'iris_shuffled.mat')


# 데이터 로드
data = loadmat(mat_file_path)
iris_data = data['iris_data'].astype(float)
iris_class = data['iris_class'].astype(int)

X1 = iris_data[0:50, :]     # 첫 50개 행 (0 ~ 49)
X2 = iris_data[50:100, :]   # 다음 50개 행 (50 ~ 99)
X3 = iris_data[100:150, :]  # 마지막 50개 행 (100 ~ 149)


# 전체 데이터 결합
X = np.vstack((X1, X2, X3))
Etrain = 0
N = X.shape[0]

def classify_KNN(K):
    curEtrain = 0
    for i in range(N):
        x = X[i, :]  # 현재 샘플
        d = np.linalg.norm(X - x, axis=1)  # 모든 샘플과의 거리

        si = np.argsort(d)  # 인덱스를 거리순으로 정렬

        # 각 클래스의 투표 수를 저장할 배열
        c = np.zeros(3)

        # 가장 가까운 K개의 이웃 확인
        for j in range(K):
            idx = si[j]
            if idx < 50:
                c[0] += 1
            elif idx < 100:
                c[1] += 1
            else:
                c[2] += 1

        # 가장 많은 투표를 받은 클래스를 선택
        maxi = np.argmax(c)

        # 실제 클래스와 비교    
        if maxi != iris_class[i]:
            curEtrain += 1
    return curEtrain
    
# 테스트할 K 값들
k_values = [5, 10, 15, 20, 25, 30]
results = [] # 결과를 저장할 리스트 (K, 오분류 수, 오분류율)

# 각 K 값에 대해 오분류 수 및 오분류율 계산
print("Calculating errors for different K values...")
for k in k_values:
    error_count = classify_KNN(k)
    error_rate = error_count / N
    results.append((k, error_count, error_rate))
    # 중간 결과 출력 (선택 사항)
    # print(f"  K={k}: Error Count = {error_count}, Error Rate = {error_rate:.4f}")

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
