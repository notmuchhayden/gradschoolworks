import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CIFAR10Loader import trainloader, testloader  # CIFAR-10 데이터 로더 import
import os


# BasicBlock: ResNet의 기본 잔차 블록 정의
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        # 첫 번째 합성곱 + 배치정규화 + ReLU
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # 두 번째 합성곱 + 배치정규화
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        # 배치 정규화 계층
        self.bn2 = nn.BatchNorm2d(planes)

        # 입력과 출력의 크기가 다를 때를 위한 downsample 계층
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        identity = x  # skip connection용 입력 저장
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity  # skip connection
        out = self.relu(out)
        return out


# 경량화된 ResNet(ResNet-18 스타일) 정의
class CIFAR10_ResNet18(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[2, 2, 2, 2], num_classes=10):
        super(CIFAR10_ResNet18, self).__init__()
        self.in_planes = 64

        # 입력 계층: 입력 채널 3(RGB), 출력 채널 32, 커널 크기 3x3, 패딩 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # 첫 번째 배치 정규화 계층
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU 활성화 함수
        self.relu = nn.ReLU(inplace=True)

        # 4개의 잔차 블록 스택 (채널 수 증가, 다운샘플링 포함)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # Global Average Pooling 및 FC 계층
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 최종 출력 클래스 수에 맞춘 완전연결 계층
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        # 잔차 블록을 쌓는 함수
        layers = []
        layers.append(block(self.in_planes, planes, stride))
        self.in_planes = planes * block.expansion
        # 나머지 블록들은 stride=1로 생성
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # 입력 계층
        # 잔차 블록 레이어 통과
        x = self.layer1(x)  # 잔차 블록 1
        x = self.layer2(x)  # 잔차 블록 2
        x = self.layer3(x)  # 잔차 블록 3
        x = self.layer4(x)  # 잔차 블록 4
        x = self.avgpool(x)  # Global Average Pooling
        x = torch.flatten(x, 1)  # 1차원 벡터로 변환
        # 완전연결 계층을 통한 클래스 예측
        x = self.fc(x)  # FC 계층
        return x


# 학습 설정 및 모델 준비
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 선택된 장치: {device}")

net = CIFAR10_ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 이어서 학습을 위한 checkpoint 불러오기
checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10_resnet18_1.pth')
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Checkpoint loaded. Resume from epoch {start_epoch}")
    else:
        net.load_state_dict(checkpoint)
        print("Model weights loaded. Training will start from epoch 0.")

# 학습 루프
num_epochs = 5  # 에폭 수는 조정 가능
for epoch in range(start_epoch, num_epochs):
    net.train()
    running_loss = 0.0
    for inputs, labels in trainloader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}")
    # 매 epoch마다 checkpoint 저장
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

# 테스트 정확도 평가
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 모델 저장
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10_resnet18_1.pth')
torch.save(net.state_dict(), save_path)
print(f"모델이 {save_path} 파일로 저장되었습니다.")
