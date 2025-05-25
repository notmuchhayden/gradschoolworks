# 11_1_MLP_classifier.m 을 Python으로 변환한 코드입니다.
# Multi-Layer Perceptron (MLP) for Classification

import numpy as np
import scipy.io as sio

def d_tanh(a):
    return (1 - a) * (1 + a)

def MLPtest(Xtst, Ttst, w, w0, v, v0):
    N = Xtst.shape[0]
    E = np.zeros((N, 1))
    Ytst = np.zeros_like(Ttst)
    
    for i in range(N):
        x = Xtst[i, :]
        t = Ttst[i, :]
        uh = np.dot(x, w) + w0
        z = np.tanh(uh)
        uo = np.dot(z, v) + v0
        y = np.tanh(uo)
        e = y - t
        E[i] = np.dot(e, e.T)
        Ytst[i, :] = [1, -1] if y[0] > y[1] else [-1, 1]
    
    SEtst = np.sum(E ** 2) / N
    diffTY = np.sum(np.abs(Ttst - Ytst), axis=1) / 2
    CEtst = np.sum(diffTY) / N
    return SEtst, CEtst

# 데이터 불러오기
data = sio.loadmat('data12_20.mat')
X = data['X']
T = data['T']

N = X.shape[0]
INP, HID, OUT = 2, 5, 2

# 가중치 초기화
w = np.random.rand(INP, HID) * 0.4 - 0.2
w0 = np.random.rand(1, HID) * 0.4 - 0.2
v = np.random.rand(HID, OUT) * 0.4 - 0.2
v0 = np.random.rand(1, OUT) * 0.4 - 0.2

eta = 0.001
Mstep = 5000
Elimit = 0.05

Serr = []
Cerr = []

for j in range(2, Mstep + 1):
    E = np.zeros((N, 1))
    for i in range(N):
        x = X[i, :]
        t = T[i, :]
        uh = np.dot(x, w) + w0
        z = np.tanh(uh)
        uo = np.dot(z, v) + v0
        y = np.tanh(uo)
        e = y - t
        E[i] = np.dot(e, e.T)

        delta_v = d_tanh(y) * e
        delta_w = d_tanh(z) * np.dot(delta_v, v.T)

        v -= eta * np.dot(z[:, np.newaxis], delta_v[np.newaxis, :])
        v0 -= eta * delta_v
        w -= eta * np.dot(x[:, np.newaxis], delta_w[np.newaxis, :])
        w0 -= eta * delta_w

    serr, cerr = MLPtest(X, T, w, w0, v, v0)
    print(f"{j} {serr:.3f} {cerr:.3f}")
    Serr.append(serr)
    Cerr.append(cerr)
    if serr < Elimit:
        break

# 학습된 가중치를 저장 (필요한 경우)
np.savez('MLPweight.npz', INP=INP, HID=HID, OUT=OUT, v=v, v0=v0, w=w, w0=w0)
