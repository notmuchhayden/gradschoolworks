import numpy as np
from scipy.io import loadmat

# 데이터 로드 (.mat 파일에 X1, X2, X3 포함되어 있어야 함)
data = loadmat('dataCh4_7.mat')
X1 = data['X1']
X2 = data['X2']
X3 = data['X3']

K = 3  # 클래스 수

# 클래스별 평균 계산
M = [np.mean(X1, axis=0), np.mean(X2, axis=0), np.mean(X3, axis=0)]

# 클래스별 공분산 계산
S = [np.cov(X1, rowvar=False), np.cov(X2, rowvar=False), np.cov(X3, rowvar=False)]

# 공통 공분산 (평균)
smean = (S[0] + S[1] + S[2]) / 3

# 학습 데이터 구성
Dtrain = np.vstack((X1, X2, X3))
Etrain = np.zeros(3)  # 세 가지 경우의 오분류 수 저장
N = X1.shape[0]       # 각 클래스의 샘플 수 (예: 100)

# 클래스별로 분류 시작
for k in range(K):
    X = Dtrain[k * N:(k + 1) * N]
    for i in range(N):
        d1 = []  # 단위 공분산 가정 (유클리드 거리)
        d2 = []  # 공통 공분산 가정 (마할라노비스 거리)
        d3 = []  # 개별 공분산 가정 (마할라노비스 거리)
        for j in range(K):
            diff = X[i] - M[j]
            d1.append(np.dot(diff, diff))
            d2.append(diff @ np.linalg.inv(smean) @ diff)
            d3.append(diff @ np.linalg.inv(S[j]) @ diff)
        
        if np.argmin(d1) != k:
            Etrain[0] += 1
        if np.argmin(d2) != k:
            Etrain[1] += 1
        if np.argmin(d3) != k:
            Etrain[2] += 1

# 오분류율 계산
Error_rate = Etrain / (N * K)
print("Error rates:")
print(f"  단위 공분산 가정      (Euclidean):      {Error_rate[0]:.4f}")
print(f"  공통 공분산 가정      (Mahalanobis):    {Error_rate[1]:.4f}")
print(f"  개별 공분산 가정      (Mahalanobis):    {Error_rate[2]:.4f}")
