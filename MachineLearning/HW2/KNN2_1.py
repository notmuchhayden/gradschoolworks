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
# iris_class를 1차원 배열로 만들고, 0-based 인덱싱 (0, 1, 2)으로 변환
iris_class = data['iris_class'].astype(int).flatten() - 1 # flatten() 및 -1 추가

# 전체 데이터
X = iris_data
N = X.shape[0] # 150

def classify_KNN(K):
    curEtrain = 0
    for i in range(N):
        x = X[i, :]  # 현재 샘플
        # 현재 샘플 x와 모든 샘플 X 간의 유클리드 거리 계산
        d = np.linalg.norm(X - x, axis=1)

        # 거리를 기준으로 인덱스 정렬 (가장 가까운 순서)
        si = np.argsort(d)

        # 각 클래스(0, 1, 2)의 투표 수를 저장할 배열 초기화
        c = np.zeros(3)

        # 가장 가까운 K개의 이웃 확인 (자기 자신 제외)
        # si[0]은 자기 자신이므로 si[1]부터 K개의 이웃을 선택
        for j in range(1, K + 1): # 0 대신 1부터 시작하고, K까지 (총 K개)
            idx = si[j] # K개의 가까운 이웃 중 j번째 이웃의 인덱스

            # --- 수정된 부분: 실제 클래스 레이블 사용 ---
            neighbor_class = iris_class[idx] # 이웃의 실제 클래스 (0, 1, 2)
            if 0 <= neighbor_class < 3: # 유효한 클래스 인덱스인지 확인
                 c[neighbor_class] += 1
            # --- 기존 방식 주석 처리 ---
            # if idx < 50:
            #     c[0] += 1
            # elif idx < 100:
            #     c[1] += 1
            # else:
            #     c[2] += 1
            # --- ---

        # 가장 많은 투표를 받은 클래스(0, 1, 2) 선택
        predicted_class = np.argmax(c)

        # --- 수정된 부분: 0-based 클래스와 비교 ---
        true_class = iris_class[i] # 현재 샘플의 실제 클래스 (0, 1, 2)
        if predicted_class != true_class:
            curEtrain += 1 # 오분류인 경우 카운트 증가
        # --- 기존 방식 주석 처리 ---
        # if maxi != iris_class[i]:
        #     curEtrain += 1
        # --- ---
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

