import numpy as np
import matplotlib.pyplot as plt

# 문제 1의 두 번째 문제 (2)

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

# 데이터 합치기 및 클래스 레이블 생성
class_data = np.vstack((class1, class2))
class_selector = np.concatenate((np.zeros(n_samples), np.ones(n_samples))) # X 를 class1, class2 로 구분하기 위한 

# PCA 구현
def pca(data, target_dimension):
    # PCA 알고리즘
    # 1. 입력 데이터 집합 𝑋의 평균 𝝁𝑥 와 공분산 Σ𝑥 를 계산
    mx = np.mean(data, axis=0)      # 데이터의 평균 계산
    data_centered = data - mx       # 데이터를 중심으로 이동 시킴
    sigmax = np.cov(data_centered, rowvar=False) # 공분산 행렬 계산
    
    # 2. 공분산 행렬에서 고유값과 고유벡터 계산
    # eigenvalues : 고유치
    # eigenvectors : 고유벡터들
    eigenvalues, eigenvectors = np.linalg.eig(sigmax)
    
    # 3. 고유치가 큰 것부터 순서대로 𝑑개의 고유치 λ1 , λ2 , ⋯ , λ𝑑 를 선택
    sorted_indices = np.argsort(eigenvalues)[::-1]          # 고유값을 기준으로 내림차순 정렬
    sorted_eigenvalues = eigenvalues[sorted_indices]        # 정렬된 고유치 생성
    sorted_eigenvectors = eigenvectors[:, sorted_indices]   # 정렬된 고유치에 해당한 정렬된 고유벡터 생성
    
    # 4. 주성분 벡터 선택
    principal_components = sorted_eigenvectors[:, :target_dimension]# 목표 차원까지의 d 개의 고유치 선택
    return principal_components, sorted_eigenvalues, mx

# LDA 구현
def lda(data, selector, target_dimension):
    # 1 .입력데이터 𝑋를 각 클래스 레이블에 따라 𝑀개의 클래스로 나누어 각각
    #    평균 𝒎𝑘 와 클래스 간 산점행렬 𝑆𝐵 , 그리고 클래스 내 산점행렬 𝑆𝑊 를 계산
    # 클래스별 평균 계산
    m1 = np.mean(data[selector == 0], axis=0)
    m2 = np.mean(data[selector == 1], axis=0)
    mk = np.mean(data, axis=0)
    # 클래스 내 분산 행렬 계산
    Sw1 = np.cov((data[selector == 0] - m1).T)
    Sw2 = np.cov((data[selector == 1] - m2).T)
    # 클래스 내 산점 행렬 Sw 계산
    Sw = Sw1 + Sw2 
    
    # 클래스 간 분산 행렬 계산
    Sb = np.outer(m1 - mk, m1 - mk) + np.outer(m2 - mk, m2 - mk)

    # 2. 고유치 분석을 통해 행렬 𝑆W^−1 𝑆𝐵 의 고유치행렬 Λ와 고유행렬벡터 𝑼를 계산
    # eigenvalues : 고유치
    # eigenvectors : 고유벡터
    eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
    
    # 3. 고유치가 큰 것부터 순서대로 𝑑개의 고유치 λ1 , λ2 , ⋯ , λ𝑑 를 선택
    sorted_indices = np.argsort(eigenvalues)[::-1]
    
    # 4. 선택한 고유치에 대응되는 고유벡터를 열벡터로 가지는 변환행렬 𝐖를 생성
    lda_vector = eigenvectors[:, sorted_indices[:target_dimension]] # 여기서는 변환 벡터 생성
    return lda_vector

# PCA 적용
pca_vector, pca_value, pca_mean = pca(class_data, target_dimension=1)

# LDA 적용
lda_vector = lda(class_data, class_selector, target_dimension=1)

# 결과 시각화
plt.figure(figsize=(8, 6))
plt.scatter(class1[:, 0], class1[:, 1], label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], label='Class 2')

# PCA 벡터 그리기 (원점에서 시작하도록 pca_mean을 시작점으로 사용)
scale_pca = 10 # 적절한 스케일링
plt.quiver(pca_mean[0], pca_mean[1], 
           pca_vector[0] * scale_pca, 
           pca_vector[1] * scale_pca, 
           angles='xy', scale_units='xy', 
           scale=1, color='r', 
           label='PCA 1st component')

# LDA 벡터 그리기 (원점에서 시작하도록 전체 평균을 시작점으로 사용)
scale_lda = 10 # 적절한 스케일링
overall_mean = np.mean(class_data, axis=0)
plt.quiver(overall_mean[0], overall_mean[1], 
           lda_vector[0] * scale_lda, 
           lda_vector[1] * scale_lda, 
           angles='xy', scale_units='xy', 
           scale=1, color='g', 
           label='LDA 1st component')

plt.xlabel('x')
plt.ylabel('y')
plt.title('PCA and LDA First Component Vectors')
plt.axis([-10, 10, -5, 10])
plt.legend()
plt.grid(True)
plt.show()