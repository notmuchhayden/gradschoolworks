pkg load statistics  % Statistics 패키지 로드

clc; clear; close all;

% 평균 및 공분산 행렬 정의
mu1 = [0; 0];
sigma1 = [10 2; 2 1];

mu2 = [0; 5];
sigma2 = [10 2; 2 1];

n = 100; % 샘플 개수

% 다변량 정규분포 난수 생성
X1 = mvnrnd(mu1, sigma1, n);
X2 = mvnrnd(mu2, sigma2, n);

% 산점도 출력
figure;
hold on;
scatter(X1(:,1), X1(:,2), 'bo', 'filled');
scatter(X2(:,1), X2(:,2), 'ro', 'filled');
axis([-10 10 -5 10]);
xlabel('X-axis');
ylabel('Y-axis');
title('Scatter Plot of Two Classes');
legend('Class 1', 'Class 2');
grid on;
hold off;

