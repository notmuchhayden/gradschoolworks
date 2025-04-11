import numpy as np
import matplotlib.pyplot as plt


# 문제 1의 첫 번째 문제 (1)

#파라미터 설정
mu1 = np.array([0, 0])
sigma1 = np.array([[4, 0], 
                   [0, 4]])

mu2 = np.array([3, 5])
sigma2 = np.array([[3, 0], 
                   [0, 5]])
n_samples = 100

# 데이터 생성
# 가우시안분포를 따르는 무작위 표본을 생성
C1 = np.random.multivariate_normal(mu1, sigma1, n_samples)
C2 = np.random.multivariate_normal(mu2, sigma2, n_samples)

# 산점도 그리기
plt.figure(figsize=(8, 6))
plt.scatter(C1[:, 0], C1[:, 1], label='C1')
plt.scatter(C2[:, 0], C2[:, 1], label='C2')
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.title('Data scatter')
plt.axis([-10, 10, -10, 15])
plt.legend()
plt.grid(True)
plt.show()






