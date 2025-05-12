import os
import numpy as np
import matplotlib.pyplot as plt

# 문제 1-(3)
# (1)에서 생성한 데이터에 K-means 알고리즘(K=3으로 설정)을 적용하여 군집화를
# 수행하시오. 이때 알고리즘 반복횟수는 10으로 한정하고, 각 반복에 따른 군집 결과 및 군
# 집의 중심을 plot을 활용하여 출력하시오. 그리고 마지막 수행 결과 출력에서는 (2)에서
# 계산하였던 평균값도 함께 출력하여 군집화를 통해 획득한 군집의 중심과 실제 데이터가
# 가진 중심 간의 차이가 어떻게 나타나는지를 관찰하시오. (최종적으로 최적의 군집화 상태
# 에 도달하기 까지 알고리즘이 몇 번 반복되었는지 언급하고, 각 반복 step마다 plot 해야
# 함)

# 학습 데이터 생성 -----------------------
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

np.random.seed(0)  # 결과 재현 가능하게 설정
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

# 데이터 병합
X = np.vstack([
    data1,
    data2,
    data3
])

# 대표벡터의 초기화 ----------------------
N, K = X.shape[0], 3
m = np.zeros((K, 2))  # 대표 벡터
Xlabel = np.zeros(N, dtype=int)  # 클러스터 할당 결과
i = 0


# 데이터 시각화
plt.figure(figsize=(8, 6))

# 색상 및 마커 지정
colors = ['b', 'r', 'g']
markers = ['d', '*', 'o']
plt.scatter(X[:, 0], X[:, 1], c='black', marker='o', alpha=0.1, label='All Data')

plt.title("Initial Data")
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
#plt.axis('equal') # x, y 축의 스케일을 동일하게 설정하여 원형 분포를 제대로 표시

# 단계 1 -- K개의 대표벡터를 선택
while i < K:
    t = np.random.randint(N)
    if not any((X[t] == m[j]).all() for j in range(i)):
        m[i] = X[t]
        plt.plot(m[i, 0], m[i, 1], 'ks')
        i += 1

plt.show()


# % K-means 반복 알고리즘의 시작 ---------------
for iteration in range(10):
    plt.figure(figsize=(8, 6))
    plt.title(f"Iteration {iteration + 1}")
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.grid(True)

    # 단계 2 -- 각 데이터를 가까운 클러스터에 할당
    for i in range(N):
        distances = np.sum((m - X[i]) ** 2, axis=1)
        Xlabel[i] = np.argmin(distances)
        plt.plot(X[i, 0], X[i, 1], colors[Xlabel[i]] + markers[Xlabel[i]], alpha=0.5)

    # 단계 3 -- 대표벡터를 다시 계산
    oldm = m.copy()
    for i in range(K):
        members = X[Xlabel == i]
        if len(members) > 0:
            m[i] = members.mean(axis=0)
        plt.plot(m[i, 0], m[i, 1], 'ks')  # 새로운 대표 벡터 표시

    # 반복 완료 조건 검사(수렴 조건 검사)
    if np.sum(np.sqrt(np.sum((oldm - m) ** 2, axis=1))) < 1e-3:
        plt.scatter(calculated_mu1[0], calculated_mu1[1], c='k', marker='x', s=150, linewidths=1, label='Calculated Mean 1')
        plt.scatter(calculated_mu2[0], calculated_mu2[1], c='k', marker='x', s=150, linewidths=1, label='Calculated Mean 2')
        plt.scatter(calculated_mu3[0], calculated_mu3[1], c='k', marker='x', s=150, linewidths=1, label='Calculated Mean 3')
        plt.show()
        break

    plt.show()
    # 단계 4 (단계 2와 3을 반복)





