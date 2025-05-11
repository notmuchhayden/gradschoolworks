import os
import numpy as np
import matplotlib.pyplot as plt

# 평균 및 공분산 정의
mu1 = np.array([0, 4])
sigma1 = np.array([[1, 0], 
                   [0, 1]])

mu2 = np.array([4, 4])
sigma2 = np.array([[1, 0], 
                   [0, 1]])

mu3 = np.array([2, 0])
sigma3 = np.array([[1, 0], 
                   [0, 1]])

# 데이터 생성
np.random.seed(0)
X1 = np.random.multivariate_normal(mu1, sigma1, 100)
X2 = np.random.multivariate_normal(mu2, sigma2, 100)
X3 = np.random.multivariate_normal(mu3, sigma3, 100)

# 평균 및 공분산
M = [np.mean(X1, axis=0), np.mean(X2, axis=0)]
S = [np.cov(X1, rowvar=False), np.cov(X2, rowvar=False)]
smean = (S[0] + S[1]) / 2  # 공통 공분산 가정