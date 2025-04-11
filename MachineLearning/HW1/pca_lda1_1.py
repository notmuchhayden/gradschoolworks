import numpy as np
import matplotlib.pyplot as plt

# 문제 1의 첫 번째 문제 (1)

# 파라미터 설정
mu1 = np.array([0, 0])
mu2 = np.array([0, 5])
sigma = np.array([[10, 2], [2, 1]]) # 공분산
n_samples = 100

# 데이터 생성
#np.random.seed(42)
# 다변량정규분포를 따르는 무작위 표본을 생성
class1 = np.random.multivariate_normal(mu1, sigma, n_samples)
class2 = np.random.multivariate_normal(mu2, sigma, n_samples)

# 산점도 그리기
plt.figure(figsize=(8, 6))
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sample Data')
plt.axis([-10, 10, -5, 10])
plt.legend()
plt.grid(True)
plt.show()