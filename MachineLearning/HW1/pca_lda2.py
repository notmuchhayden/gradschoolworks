import numpy as np
import h5py
import matplotlib.pyplot as plt


def pca(data):
    # PCA 알고리즘
    # 1. 입력 데이터 집합 𝑋의 평균 𝝁𝑥 와 공분산 Σ𝑥 를 계산
    mx = np.mean(data, axis=0)      # 데이터의 평균 계산
    data_centered = data - mx       # 데이터를 중심으로 이동 시킴
    sigmax = np.cov(data_centered, rowvar=False) # 공분산 행렬 계산
    
    # 2. 공분산 행렬에서 고유값과 고유벡터 계산
    # eigenvalues : 고유치
    # eigenvectors : 고유벡터들
    eigenvalues, eigenvectors = np.linalg.eigh(sigmax)
    
    # 3. 고유치가 큰 것부터 순서대로 𝑑개의 고유치 λ1 , λ2 , ⋯ , λ𝑑 를 선택
    sorted_indices = np.argsort(eigenvalues)[::-1]          # 고유값을 기준으로 내림차순 정렬
    sorted_eigenvalues = eigenvalues[sorted_indices]        # 정렬된 고유치 생성
    sorted_eigenvectors = eigenvectors[:, sorted_indices]   # 정렬된 고유치에 해당한 정렬된 고유벡터 생성
    
    # 4. 문제1 의 구현과는 다르게 주성분 벡터 선택은 pca 함수 밖에서 수행
    return sorted_eigenvectors, sorted_eigenvalues, mx

# class 가 2개 이상인 경우에 대응하도록 확장
def lda(data, selector):
    # 1 .입력데이터 𝑋를 각 클래스 레이블에 따라 𝑀개의 클래스로 나누어 각각
    #    평균 𝒎𝑘 와 클래스 간 산점행렬 𝑆𝐵 , 그리고 클래스 내 산점행렬 𝑆𝑊 를 계산
    # 클래스별 평균 계산
    class_labels = np.unique(selector)
    ms = [np.mean(data[selector == label], axis=0) for label in class_labels]
    mk = np.mean(data, axis=0)
    
    # 클래스 내 분산 행렬 계산
    Sw = np.zeros((data.shape[1], data.shape[1]))
    for label, mean in zip(class_labels, ms):
        X_class = data[selector == label]
        Sw += np.cov(X_class, rowvar=False) * (X_class.shape[0] - 1)

    # 클래스 간 분산 행렬 계산
    Sb = np.zeros((data.shape[1], data.shape[1]))
    for label, mean in zip(class_labels, ms):
        n = np.sum(selector == label)
        mean_diff = (mean - mk).reshape(-1, 1)
        Sb += n * (mean_diff @ mean_diff.T)

    # 2. 고유치 분석을 통해 행렬 𝑆W^−1 𝑆𝐵 의 고유치행렬 Λ와 고유행렬벡터 𝑼를 계산
    # eigenvalues : 고유치
    # eigenvectors : 고유벡터
    eigenvalues, eigenvectors = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)
    
    # 3. 고유치가 큰 것부터 순서대로 𝑑개의 고유치 λ1 , λ2 , ⋯ , λ𝑑 를 선택
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # 4. 선택한 고유치에 대응되는 고유벡터를 열벡터로 가지는 변환행렬 𝐖를 생성
    lda_vectors = eigenvectors[:, sorted_indices].real  # 복소수 부분 제거
    return lda_vectors

# 데이터 로드 (h5py 사용)
with h5py.File('HW1_COIL20.mat', 'r') as mat_data:
    X = np.array(mat_data['X']).T  # 전치하여 형식 맞추기
    Y = np.array(mat_data['Y']).ravel()

# PCA를 통한 2차원 특징 추출
eigenvectors, eigenvalues, pca_mean = pca(X)
X_centered = X - pca_mean
X_pca = np.dot(X_centered, eigenvectors[:, :2]) # 주성분벡터 2번째까지 선택
X_pca[:, 0] *= -1  # 좌우 반전
X_pca[:, 1] *= -1  # 상하 반전

# PCA 결과 시각화
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for label in np.unique(Y):
    plt.scatter(X_pca[Y == label, 0], X_pca[Y == label, 1], label=f'Class {label}', alpha=0.6)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("PCA 2D Projection")

# PCA 95% 정보 보존 후 LDA 적용
eigenvectors, eigenvalues, pca_mean = pca(X)
cumulative_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)
n_components = np.searchsorted(cumulative_variance, 0.95) + 1
X_centered = X - pca_mean
X_pca_95 = np.dot(X_centered, eigenvectors[:, :n_components])

lda_vectors = lda(X_pca_95, Y)
X_lda = np.dot(X_pca_95, lda_vectors[:, :2])
X_lda[:, 0] *= -1  # LDA 결과 좌우 반전
#X_lda[:, 1] *= -1  # LDA 결과 상하 반전

# LDA 결과 시각화
plt.subplot(1, 2, 2)
for label in np.unique(Y):
    plt.scatter(X_lda[Y == label, 0], X_lda[Y == label, 1], label=f'Class {label}', alpha=0.6)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.title("LDA 2D Projection")

plt.show()
