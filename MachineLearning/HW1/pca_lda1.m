% 데이터의 생성 --------------------------------------
N=100;
m1=[0 0]; s1=[9 0;0 1];
m2=[0 4]; s2=[9 0;0 1];
X1=randn(N,2)*sqrtm(s1)+repmat(m1,N,1); % 클래스1 데이터 생성
X2=randn(N,2)*sqrtm(s2)+repmat(m2,N,1); % 클래스2 데이터 생성
figure(1);
plot(X2(:,1),X2(:,2),'ro'); hold on
plot(X1(:,1),X1(:,2),'*');
axis([-10 10 -5 10]);
save data8_9 X1 X2

% PCA에 의한 분석 ------------------------------------
X=[X1;X2];
M=mean(X); S=cov(X); % 평균과 공분산 계산
[V,D]=eig(S); % 고유치분석(U:고유벡터행렬, D: 고유치행렬)
w1=V(:,2); % 첫 번째 주성분벡터
figure(1); % 첫 번째 주성분벡터를 공간에 표시
line([0 w1(1)*D(2,2)]+M(1),[0 w1(2)*D(2,2)]+M(2));
YX1=w1'*X1'; YX2=w1'*X2'; % 첫 번째 주성분벡터로 사영
pYX1=w1*YX1; pYX2=w1*YX2; % 사영된 데이터를 2차원 공간으로 환원
figure(2); % 사영된 데이터를 2차원 공간으로 환원하여 표시
plot(pYX1(1,:), pYX1(2,:)+M(2),'*');
hold on axis([-10 10 -5 10]);
plot(pYX2(1,:), pYX2(2,:)+M(2),'ro');

% LDA에 의한 분석 -----------------------------------
m1=mean(X1); m2=mean(X2);
Sw=N*cov(X1)+N*cov(X2); % within scatter 계산
Sb=(m1-m2)'*(m1-m2); % between scatter 계산
[V,D]=eig(Sb*inv(Sw)); % 고유치 분석
w=V(:,2); % 찾아진 벡터
figure(1); % 찾아진 벡터를 공간에 표시
line([0 w(1)*-8]+M(1),[0 w(2)*-8]+M(2));
YX1=w'*X1'; YX2=w'*X2'; % 벡터w 방향으로 사영
pYX1=w*YX1; pYX2=w*YX2; % 사영된 데이터를 2차원 공간으로 환원
figure(2); % 사영된 데이터를 2차원 공간으로 환원하여 표시
plot(pYX1(1,:)+M(1), pYX1(2,:),'*');
plot(pYX2(1,:)+M(1), pYX2(2,:),'ro');
