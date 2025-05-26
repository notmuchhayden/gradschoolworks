# 1. (100점) 참고교재의 프로그램 11-1을 참고하여, 주어진 데이터 HW4data을 이용하여 다층
# 퍼셉트론을 구현하시오. 데이터의 변수 X는 400개의 2차원 샘플로 구성되었으며, T는 400
# 개 데이터에 대한 클래스 레이블로 1과 –1로 구분되어 있음.
# (단, 종료조건으로 최대 반복횟수는 1,000회 그리고 학습오차 0.05 미만으로 설정하시오)

# (3) (20점) 1.(2)에서 학습된 신경망의 결정경계를 학습 데이터와 함께 그래프로 표시하시오.
# (※ 결정경계 계산 시 meshgrid 범위:
# [x, y] = meshgrid([-4:0.1:10], [-4:0.1:10]);)

import os
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# 입력 벡터를 받아서 하이퍼탄젠트 함수의 1차 도함수의 값을 반환
def d_tanh(a):
    return (1 - a) * (1 + a)

# 데이터 집합과 가중치를 받아 평균제곱오차와 분류오차를 계산
def MLPtest(Xtst, Ttst, w, w0, v, v0):
    N = Xtst.shape[0]                       # 데이터의 수
    E = np.zeros((N, 1))
    Ytst = np.zeros_like(Ttst)              
    for i in range(N):                      # 각 데이터에 대한 인식 시작
        x, t = Xtst[i, :], Ttst[i, :]       # 입/출력 데이터 설정        
        uh = np.dot(x, w) + w0              # 은닉 뉴런의 가중합 계산
        z = np.tanh(uh)                     # 은닉 뉴런의 출력 계산
        uo = np.dot(z, v) + v0              # 출력 뉴런의 가중합 계산
        y = np.tanh(uo)                     # 출력 뉴런의 출력 계산
        e = y - t                           # 출력 오차 계산
        E[i] = np.dot(e, e.T)               # 제곱오차 계산        
        if y[0, 0] > 0:
            Ytst[i, 0] = 1                  # 최종 인식 결과 판단
        else:
            Ytst[i, 0] = -1
    SEtst = np.sum(E) / N                   # 평균제곱오차 계산 (수정: E는 이미 각 샘플의 제곱오차임)
    diffTY = np.sum(np.abs(Ttst - Ytst), axis=1) / 2
    CEtst = np.sum(diffTY) / N              # 분류오차 계산
    return SEtst, CEtst

# 데이터 불러오기
script_dir = os.path.dirname(os.path.abspath(__file__))
mat_file_path = os.path.join(script_dir, 'HW4data_Python.mat')
data = sio.loadmat(mat_file_path)

X = data['X'] # 2차원 샘플 데이터 (400, 2)
T = data['T'] # 1차원 클래스 레이블 (400, 1)

# 다층 퍼셉트론 구성
N = X.shape[0]                              # 데이터의 수
INP, HID, OUT = 2, 5, 1                     # 입력 뉴런 수, 은닉 뉴런 수, 출력 뉴런 수

# 가중치 초기화
w = np.random.rand(INP, HID) * 0.4 - 0.2    # 입력->은닉 뉴런 가중치 초기화
w0 = np.random.rand(1, HID) * 0.4 - 0.2     # 바이어스
v = np.random.rand(HID, OUT) * 0.4 - 0.2    # 은닉->출력 뉴런 가중치 초기화
v0 = np.random.rand(1, OUT) * 0.4 - 0.2

eta = 0.001                                 # 학습률 설정
Mstep = 1000                                # 반복횟수 설정
Elimit = 0.05

Serr = []
Cerr = []

for j in range(1, Mstep + 1):               # 학습 반복 시작 (수정: 1부터 Mstep회 반복하도록 변경)
    E = np.zeros((N, 1))
    for i in range(N):                      # 각 데이터에 대한 반복 시작
        x, t = X[i, :], T[i, :]             # 입력과 목표 출력 데이터 선택
        uh = np.dot(x, w) + w0              # 은닉 뉴런의 가중합 계산
        z = np.tanh(uh)                     # 은닉 뉴런의 출력 계산
        uo = np.dot(z, v) + v0              # 출력 뉴런의 가중합 계산
        y = np.tanh(uo)                     # 출력 뉴런의 출력 계산
        e = y - t                           # 출력 오차 계산
        E[i] = np.dot(e, e.T)               # 제곱오차 계산
        delta_v = d_tanh(y) * e             # 학습을 위한 델타값 계산 (출력뉴런)
        delta_w = d_tanh(z) * (delta_v @ v.T) # 학습을 위한 델타값 계산 (은닉뉴런)
        v -= eta * (z.T @ delta_v)          # 출력 뉴런의 가중치 수정        
        v0 -= eta * delta_v                 # 출력 뉴런의 바이어스 가중치 수정
        w -= eta * (x.reshape(-1, 1) @ delta_w) # 은닉 뉴런의 가중치 수정
        w0 -= eta * delta_w                 # 은닉 뉴런의 바이어스 가중치 수정

    serr, cerr = MLPtest(X, T, w, w0, v, v0) # 학습 오차 계산 함수 호출
    print(f"{j} {serr:.3f} {cerr:.3f}")     # 오차 변화 출력
    Serr.append(serr)                       # 오차 변화 저장
    Cerr.append(cerr)
    if serr < Elimit:
        break

# (3) (20점) 학습된 신경망의 결정경계를 학습 데이터와 함께 그래프로 표시
# (※ 결정경계 계산 시 meshgrid 범위: [x, y] = meshgrid([-4:0.1:10], [-4:0.1:10]);)
print("\n문제 (3): 결정경계 및 학습 데이터 시각화 중...")

# Meshgrid 생성
h_mesh = 0.1  # meshgrid 간격
x_mesh_min, x_mesh_max = -4, 10
y_mesh_min, y_mesh_max = -4, 10
# np.arange는 마지막 값을 포함하지 않으므로, x_mesh_max + h_mesh 와 같이 설정
xx, yy = np.meshgrid(np.arange(x_mesh_min, x_mesh_max + h_mesh, h_mesh),
                     np.arange(y_mesh_min, y_mesh_max + h_mesh, h_mesh))

# Meshgrid 위의 각 점에 대한 예측 수행 (학습된 가중치 w, w0, v, v0 사용)
grid_points = np.c_[xx.ravel(), yy.ravel()] # 모든 grid 점들을 1차원으로 펼쳐서 (N_points, 2) 형태로 만듦
uh_grid = np.dot(grid_points, w) + w0       # 은닉층 가중합
z_grid = np.tanh(uh_grid)                   # 은닉층 출력
uo_grid = np.dot(z_grid, v) + v0            # 출력층 가중합
y_grid_pred = np.tanh(uo_grid)              # 출력층 출력 (예측값)

# 예측값을 클래스 레이블로 변환 (y > 0 이면 1, 아니면 -1)
Z_boundary = np.where(y_grid_pred.reshape(xx.shape) > 0, 1, -1)

# 결정경계 및 학습 데이터 플롯
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z_boundary, cmap=plt.cm.RdBu, alpha=0.7) # 결정경계 등고선 플롯

# 학습 데이터 산점도 (기존 산점도 코드와 유사)
X_class1_plot = X[T[:, 0] == 1]
X_class_minus1_plot = X[T[:, 0] == -1]
plt.scatter(X_class1_plot[:, 0], X_class1_plot[:, 1], c='blue', marker='o', edgecolor='k', label='Class 1')
plt.scatter(X_class_minus1_plot[:, 0], X_class_minus1_plot[:, 1], c='red', marker='x', edgecolor='k', label='Class -1')

plt.title('Decision Boundary with Training Data (Problem 3)')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True)
plt.xlim(xx.min(), xx.max()) # 그래프 범위를 meshgrid에 맞춤
plt.ylim(yy.min(), yy.max())
plt.show()

