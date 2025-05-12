import os
import numpy as np
import h5py # Changed from scipy.io import loadmat
import matplotlib.pyplot as plt

# 문제 2-(2)
# 참고교재의 프로그램 10-1을 참고하여, 주어진 데이터에 대한 가우시안 혼합 모델
# 을 적용하여 분석을 수행하시오. 이때 가우시안 성분의 수는 {2, 6, 10} 으로 변경해 가며
# 수행하고, 각각의 최종 결과를 산점도로 출력하여 이를 비교한 후 그 결과를 서술하시오.

# 현재 파일 경로 기준으로 .mat 파일 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(script_dir, 'HW3Data.mat')

# 데이터 로드 (h5py 사용으로 변경)
with h5py.File(mat_file_path, 'r') as f:
    hw3data = np.array(f['data']).astype(float)


def gausspdf(X, mu, sigma):    
    n = X.shape[0]  # 입력 벡터의 차원 (dims)
    N = X.shape[1]  # 데이터의 수

    # mu를 브로드캐스팅을 위해 열 벡터 (dims x 1) 형태로 변환
    # 입력 mu가 (dims,) 또는 (dims, 1) 형태여도 동일하게 작동
    mu_col = mu.reshape(-1, 1)

    # 차이 계산: X - mu (브로드캐스팅 사용) -> (dims x N)
    diff = X - mu_col

    try:
        # sigma의 역행렬 및 행렬식 계산
        inv_sigma = np.linalg.inv(sigma)
        det_sigma = np.linalg.det(sigma)

        # 행렬식이 0 또는 음수인지 확인 (수치적 안정성)
        if det_sigma <= 1e-12: # 매우 작은 양수 임계값
            print(f"Warning: det(sigma) (~{det_sigma:.2e}) is close to zero or negative. Returning zero probability.")
            # 0에 가까운 확률 또는 0 반환 (상황에 따라 결정)
            return np.zeros(N) + 1e-100

        # 정규화 상수 계산
        norm_factor = 1.0 / (np.power(2 * np.pi, n / 2) * np.sqrt(det_sigma))
        # 또는: coeff = 1.0 / ((np.sqrt(2 * np.pi))**n * np.sqrt(det_sigma))

        # 마할라노비스 거리 제곱 계산 (효율적인 방식)
        # (X-Mu)^T * inv(sigma) * (X-Mu) 의 대각 성분과 동일
        # diff: (n, N), inv_sigma: (n, n)
        # inv_sigma @ diff -> (n, N)
        # diff * (inv_sigma @ diff) -> element-wise (n, N)
        # np.sum(..., axis=0) -> sum over dimensions -> (N,)
        mahal_sq = np.sum(diff * (inv_sigma @ diff), axis=0)

        # 확률 밀도 값 계산
        out = norm_factor * np.exp(-0.5 * mahal_sq)

        # 언더플로우 방지 (결과가 0이 되는 것을 막기 위해)
        out[out < 1e-100] = 1e-100 # 필요에 따라 이 값 조정

    except np.linalg.LinAlgError:
        # 역행렬 계산 중 특이 행렬 오류 처리
        print(f"Warning: Singular matrix encountered during inversion. Returning zero probability.")
        return np.zeros(N) + 1e-100 # 0에 가까운 확률 또는 0 반환
    return out

def drawgraph(X, Mu, Sigma, cnt):
    M = Mu.shape[0]
    plt.figure(cnt)
    plt.plot(X[0, :], X[1, :], '*')
    plt.grid(True)
    #plt.xlim(-0.5, 5.5)
    #plt.ylim(-0.5, 3.5)
    plt.plot(Mu[:, 0], Mu[:, 1], 'r*')
    t = np.linspace(-np.pi, np.pi, 100)
    for j in range(M):
        A = np.sqrt(2) * np.column_stack((np.cos(t), np.sin(t)))
        ell = A @ np.linalg.cholesky(Sigma[j]) + Mu[j]
        plt.plot(ell[:, 0], ell[:, 1], 'r-', linewidth=2)

# ---- 데이터 불러오기 ----
X = hw3data         # X: 학습 데이터
N = X.shape[1]      # N: 데이터의 수
M = 2               # M: 가우시안 성분의 수

# ---- 파라미터 초기화 ----
Mu = np.random.rand(M, 2) * 5                   # 파라미터의 초기화(평균)
Sigma = np.array([np.eye(2) for _ in range(M)]) # 파라미터의 초기화(분산)
alpha = np.ones(M) * (1/M)                      # 파라미터의 초기화(혼합계수)

drawgraph(X, Mu, Sigma, 1)

# ---- EM 알고리즘 ----
Maxtau = 100
L = []

for tau in range(1, Maxtau + 1):
    # E-step
    px = np.zeros((M, N))
    for j in range(M):
        px[j, :] = gausspdf(X, Mu[j], Sigma[j])
    sump = np.dot(alpha, px)
    r = (alpha[:, np.newaxis] * px).T / sump[:, np.newaxis]

    # 로그우도 계산
    L.append(np.sum(np.log(sump)))

    # M-step
    for j in range(M):
        rj = r[:, j]
        sumr = np.sum(rj)
        Rj = rj.reshape(1, -1)
        Mu[j] = (X @ rj) / sumr
        diff = X - Mu[j].reshape(2, 1)
        Sigma[j] = (diff * Rj) @ diff.T / sumr
        alpha[j] = sumr / N

    if tau % 10 == 1:
        drawgraph(X, Mu, Sigma, tau // 10 + 1)

drawgraph(X, Mu, Sigma, Maxtau + 1)

#plt.figure(Maxtau + 2)
plt.figure(figsize=(8, 6))
plt.plot(L)
plt.title("Log-Likelihood")
plt.xlabel("Iteration")
plt.ylabel("Log-Likelihood")
plt.grid(True)
plt.show()
