import os
import numpy as np
import matplotlib.pyplot as plt

# 문제 1-(2)
# (1)에서 생성한 데이터의 각 클래스의 평균을 계산하고, 이를 (1)의 평면에 함께 표
# 시하시오.

# 평균 1 (데이터 생성 시 사용)
mu1_orig = np.array([0, 4])
# 공분산 1
sigma1 = np.array([[1, 0], 
                   [0, 1]])
# 평균 2 (데이터 생성 시 사용)
mu2_orig = np.array([4, 4])
# 공분산 2
sigma2 = np.array([[1, 0], 
                   [0, 1]])
# 평균 3 (데이터 생성 시 사용)
mu3_orig = np.array([2, 0])
# 공분산 3
sigma3 = np.array([[1, 0], 
                   [0, 1]])

# 각 클래스당 생성할 데이터 개수
num_samples_per_class = 100

# 클래스 1 데이터 생성
data1 = np.random.multivariate_normal(mu1_orig, sigma1, num_samples_per_class)
# 클래스 2 데이터 생성
data2 = np.random.multivariate_normal(mu2_orig, sigma2, num_samples_per_class)
# 클래스 3 데이터 생성
data3 = np.random.multivariate_normal(mu3_orig, sigma3, num_samples_per_class)

# 생성된 데이터로부터 각 클래스의 평균 계산
calculated_mu1 = np.mean(data1, axis=0)
calculated_mu2 = np.mean(data2, axis=0)
calculated_mu3 = np.mean(data3, axis=0)

# 데이터 플롯
plt.figure(figsize=(8, 6))
plt.scatter(data1[:, 0], data1[:, 1], c='r', marker='d', alpha=0.5, label='Class 1 (mu1, sigma1)')
plt.scatter(data2[:, 0], data2[:, 1], c='g', marker='*', alpha=0.5, label='Class 2 (mu2, sigma2)')
plt.scatter(data3[:, 0], data3[:, 1], c='b', marker='o', alpha=0.5, label='Class 3 (mu3, sigma3)')

# 계산된 평균 지점 표시
plt.scatter(calculated_mu1[0], calculated_mu1[1], c='r', marker='x', s=150, linewidths=3, label='Calculated Mean 1')
plt.scatter(calculated_mu2[0], calculated_mu2[1], c='g', marker='x', s=150, linewidths=3, label='Calculated Mean 2')
plt.scatter(calculated_mu3[0], calculated_mu3[1], c='b', marker='x', s=150, linewidths=3, label='Calculated Mean 3')

plt.title('Generated 2D Gaussian Data with Calculated Means')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
#plt.axis('equal') # x, y 축의 스케일을 동일하게 설정하여 원형 분포를 제대로 표시
plt.show()

# 계산된 평균 출력 (확인용)
print("Calculated Mean for Class 1:", calculated_mu1)
print("Original Mean for Class 1:", mu1_orig)
print("Calculated Mean for Class 2:", calculated_mu2)
print("Original Mean for Class 2:", mu2_orig)
print("Calculated Mean for Class 3:", calculated_mu3)
print("Original Mean for Class 3:", mu3_orig)
