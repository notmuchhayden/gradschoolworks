import torch
import torchvision
import torchvision.transforms as transforms
import os

# 현재 파일 위치 기준 data 폴더 경로 생성
data_dir = os.path.join(os.path.dirname(__file__), 'data')

# 데이터 전처리: 이미지를 Tensor로 변환하고 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
