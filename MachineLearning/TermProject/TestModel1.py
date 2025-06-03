import torch
from torch import nn, optim

x = torch.randn(100, 1)  # 100개의 샘플, 1차원 입력
y = 4 * x + 2 + torch.randn(100, 1) * 0.1  # y = 3x + 2 + noise

model = nn.Linear(1, 1)  # 단순 선형 회귀 모델
criterion = nn.MSELoss()  # 손실 함수: 평균 제곱 오차
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 확률적 경사 하강법

for epoch in range(1000):  # 100 에폭 동안 학습
    output = model(x)  # 모델 예측
    cost = criterion(output, y)  # 손실 계산
    
    optimizer.zero_grad()  # 기울기 초기화
    cost.backward()  # 기울기 계산
    optimizer.step()  # 가중치 업데이트
    
    if (epoch + 1) % 100 == 0:  # 100 에폭마다 출력
        print(f'Epoch [{epoch + 1:4d}/1000], Model: {list(model.parameters())}, Loss: {cost.item():.4f}')