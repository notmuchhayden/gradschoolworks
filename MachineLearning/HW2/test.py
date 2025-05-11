import numpy as np

# X가 150xN 형태의 데이터라고 가정합니다. (N은 특징의 수)
# 처음 50개는 클래스 0, 다음 50개는 클래스 1, 마지막 50개는 클래스 2라고 가정합니다.
# 예시 데이터 생성 (실제 데이터 X를 사용하세요)
# num_features = 4 # Iris 데이터의 특징 수
# X = np.vstack([
#     np.random.rand(50, num_features) + [0, 0, 0, 0], # 클래스 0 데이터
#     np.random.rand(50, num_features) + [1, 1, 1, 1], # 클래스 1 데이터
#     np.random.rand(50, num_features) + [2, 2, 2, 2]  # 클래스 2 데이터
# ])

# 각 클래스별 데이터 분할 인덱스 정의
class_size = 50
train_size_per_class = 40
test_size_per_class = 10

train_indices = []
test_indices = []

for i in range(3): # 3개의 클래스에 대해 반복
    start_index = i * class_size
    # 훈련 데이터 인덱스: 각 클래스의 처음 40개
    train_indices.extend(range(start_index, start_index + train_size_per_class))
    # 테스트 데이터 인덱스: 각 클래스의 나머지 10개
    test_indices.extend(range(start_index + train_size_per_class, start_index + class_size))

print(train_indices)
print(test_indices)

# 인덱스를 사용하여 데이터 분할
#train_data = X[train_indices, :]
#test_data = X[test_indices, :]

# 결과 확인 (선택 사항)
#print(f"원본 데이터 형태: {X.shape}")
#print(f"훈련 데이터 형태: {train_data.shape}") # 예상: (120, N)
#print(f"테스트 데이터 형태: {test_data.shape}") # 예상: (30, N)

# train_data에는 각 클래스에서 40개씩 총 120개의 데이터가 포함됩니다.
# test_data에는 각 클래스에서 10개씩 총 30개의 데이터가 포함됩니다.
