import os
import numpy as np
import matplotlib.pyplot as plt

# 문제 1-(1)
# (데이터 생성) 다음과 같은 평균과 공분산을 가지는 2차원 데이터를 클래스 당 100
# 개씩 생성하고, 이를 2차원 평면에 표시하시오(총 3개의 클래스를 생성하고, 클래스별로
# 색을 다르게 해서 plot할 것).

# 평균 1
mu1 = np.array([0, 4])
# 공분산 1
sigma1 = np.array([[1, 0], 
                   [0, 1]])
# 평균 2 
mu2 = np.array([4, 4])
# 공분산 2
sigma2 = np.array([[1, 0], 
                   [0, 1]])
# 평균 3 (주석 수정: 평균 2 -> 평균 3)
mu3 = np.array([2, 0])
# 공분산 3 (주석 수정: 공분산 2 -> 공분산 3)
sigma3 = np.array([[1, 0], 
                   [0, 1]])

# 각 클래스당 생성할 데이터 개수
num_samples_per_class = 100

# 클래스 1 데이터 생성
data1 = np.random.multivariate_normal(mu1, sigma1, num_samples_per_class)
# 클래스 2 데이터 생성
data2 = np.random.multivariate_normal(mu2, sigma2, num_samples_per_class)
# 클래스 3 데이터 생성
data3 = np.random.multivariate_normal(mu3, sigma3, num_samples_per_class)

# 데이터 플롯
plt.figure(figsize=(8, 6))
plt.scatter(data1[:, 0], data1[:, 1], c='r', marker='d', alpha=0.5, label='Class 1 (mu1, sigma1)')
plt.scatter(data2[:, 0], data2[:, 1], c='g', marker='*', alpha=0.5, label='Class 2 (mu2, sigma2)')
plt.scatter(data3[:, 0], data3[:, 1], c='b', marker='o', alpha=0.5, label='Class 3 (mu3, sigma3)')

plt.title('Generated 2D Gaussian Data (100 samples per class)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
#plt.axis('equal') # x, y 축의 스케일을 동일하게 설정하여 원형 분포를 제대로 표시
plt.show()

