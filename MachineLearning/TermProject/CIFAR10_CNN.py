import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from CIFAR10Loader import trainloader, testloader
import os

# LeNet-5 스타일의 CNN 모델 정의
class CIFAR10_CNN(nn.Module):
    def __init__(self):
        super(CIFAR10_CNN, self).__init__()
        # LeNet-5 스타일의 기본 CNN 구조
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, stride=1, padding=2)  # 32x32x3 -> 32x32x6
        self.pool = nn.AvgPool2d(2, 2)  # 평균 풀링
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)  # 16x16x6 -> 12x12x16
        # FC 계층
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.pool(self.conv1(x)))  # 32x32x3 -> 32x32x6 -> 16x16x6
        x = torch.tanh(self.pool(self.conv2(x)))  # 16x16x6 -> 12x12x16 -> 6x6x16
        x = x.view(x.size(0), -1)  # 자동 flatten
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


# device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"현재 선택된 장치: {device}")

# 모델, 손실 함수, 옵티마이저 정의
net = CIFAR10_CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

# 이어서 학습을 위한 checkpoint 불러오기
checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10_cnn.pth')
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
num_epochs = 5
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
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(trainloader):.4f}")
    # 매 epoch마다 checkpoint 저장
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

# 정확도 측정
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
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cifar10_cnn.pth')
torch.save(net.state_dict(), save_path)
print(f"모델이 {save_path} 파일로 저장되었습니다.")
