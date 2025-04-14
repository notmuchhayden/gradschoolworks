import numpy as np
import matplotlib.pyplot as plt

# 평균 및 공분산 정의
mu1 = np.array([0, 0])
sigma1 = np.array([[4, 0], 
                   [0, 4]])

mu2 = np.array([3, 5])
sigma2 = np.array([[3, 0], 
                   [0, 5]])

# 데이터 생성
np.random.seed(0)
X1 = np.random.multivariate_normal(mu1, sigma1, 100)
X2 = np.random.multivariate_normal(mu2, sigma2, 100)

# 평균 및 공분산
M = [np.mean(X1, axis=0), np.mean(X2, axis=0)]
S = [np.cov(X1, rowvar=False), np.cov(X2, rowvar=False)]
smean = (S[0] + S[1]) / 2  # 공통 공분산 가정

# 새로운 데이터
x1 = np.array([1, 2])
x2 = np.array([0, -2])
X_new = [x1, x2]

# 판별 함수 정의
def classify_common_cov(x, M, smean):
    # 클래스 공통 공분산 행렬 : (x - m)^T * sigma^-1 * (x - m)
    d = [ (x - m).T @ np.linalg.inv(smean) @ (x - m) for m in M ]
    return np.argmin(d)

def classify_general_cov(x, M, S):
    # 일반적인 공분산 행렬 : (x - m)^T * sigma^-1 * (x - m) + ln|sigma|
    d = [ (x - m).T @ np.linalg.inv(s) @ (x - m) for m, s in zip(M, S) ]
    return np.argmin(d)

# 결과 출력
for i, x in enumerate(X_new):
    c_common = classify_common_cov(x, M, smean)
    c_indiv = classify_general_cov(x, M, S)
    print(f"x{i+1} = {x} => 공통공분산: 클래스 {c_common+1}, 일반공분산: 클래스 {c_indiv+1}")

# 산점도 그리기
plt.figure(figsize=(8, 6))
plt.scatter(X1[:, 0], X1[:, 1], label='Class 1', alpha=0.6)
plt.scatter(X2[:, 0], X2[:, 1], label='Class 2', alpha=0.6)
plt.scatter(x1[0], x1[1], marker='x', color='red', s=100, label='x1 = [1, 2]')
plt.scatter(x2[0], x2[1], marker='^', color='green', s=100, label='x2 = [0, -2]')
plt.legend()
plt.title("Generated Gaussian Data with New Samples")
plt.xlabel("x1")
plt.ylabel("x2")
plt.grid(True)
plt.axis('equal')
plt.show()
