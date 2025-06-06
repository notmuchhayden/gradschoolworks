import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CIFAR10Loader import trainloader, testloader  # trainloader와 testloader를 CIFAR10Loader.py에서 import
import os


# 1. 신경망 모델 설계
# CIFAR-10 데이터셋 분류를 위한 CNN 모델 정의
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        # 첫 번째 합성곱 계층: 입력 채널 3(RGB), 출력 채널 32, 커널 크기 3x3, 패딩 1
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 두 번째 합성곱 계층: 입력 채널 32, 출력 채널 64, 커널 크기 3x3, 패딩 1
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 최대 풀링 계층: 커널 크기 2x2, 스트라이드 2
        self.pool = nn.MaxPool2d(2, 2)
        # 첫 번째 완전연결 계층: 입력 64*8*8, 출력 256
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        # 두 번째 완전연결 계층: 입력 256, 출력 10(클래스 수)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # 첫 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv1(x)))
        # 두 번째 합성곱 + ReLU + 풀링
        x = self.pool(F.relu(self.conv2(x)))
        # 피처맵을 1차원 벡터로 변환
        x = x.view(-1, 64 * 8 * 8)
        # 완전연결 계층 + ReLU
        x = F.relu(self.fc1(x))
        # 출력 계층 (클래스별 점수)
        x = self.fc2(x)
        return x


# device 설정 (GPU 사용 가능 시 GPU, 아니면 CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 선택된 장치: {device}")

# 2. 손실 함수와 최적화 도구 정의
net = CIFAR10_CNN().to(device)  # 모델을 device로 이동
criterion = nn.CrossEntropyLoss()  # 손실 함수: 교차 엔트로피
optimizer = optim.Adam(net.parameters(), lr=0.001)  # 최적화 알고리즘: Adam, 학습률 0.001

# 3. 모델 학습 함수 정의
for epoch in range(20):  # 에폭 수는 자유롭게 조정
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 device로 이동
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader)}")



correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)  # 데이터를 device로 이동
        outputs = net(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {100 * correct / total:.2f}%')

# 학습된 모델 저장
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10_cnn.pth')
torch.save(net.state_dict(), save_path)
print(f"모델이 {save_path} 파일로 저장되었습니다.")
