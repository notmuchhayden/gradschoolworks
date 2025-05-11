load data10_2       % 데이터 불러오기
X=data';            % X: 학습 데이터
N=size(X,2);        % N: 데이터의 수
M=6;                % M: 가우시안 성분의 수
Mu=rand(M,2)*5;     % 파라미터의 초기화(평균)
for i=1:M           % 파라미터의 초기화(분산)
    Sigma(i,1:2,1:2) = [1 0; 0 1];
end
alpha=zeros(6,1)+1/6; % 파라미터의 초기화(혼합계수)
drawgraph(X, Mu, Sigma, 1); % 그래프그리기 함수 호출

Maxtau=100;         % 최대 반복 횟수
for tau=1:Maxtau
    % E-step --------------------------
    for j=1:M       % p(xi|uj,qj^2) 계산
        px(j,:) = gausspdf(X, Mu(j,:), reshape(Sigma(j,:,:), 2, 2));
    end
    sump=px'*alpha      % aj * p(xi | mj, qj^2) 계산
    for j=1:M           % rij 계산
        r(:,j) = (alpha(j)*px(j,:))'./sump;
    end
    L(tau)=sum(log(sump)); % 현재 파라미터의 로그우도 계산

    % M-step --------------------------
    for j=1:M
        sumr=sum(r(:,j))    % rij 의 성분별 합산
        Rj=repmat(r(:,j),1,2)'; % 행렬 계산을 위한 준비
        Mu(j,:)=sum(Rj.*X,2)/sumr; % 새로운 평균
        % 새로운 공분산 계산
        rxmu= (X-repmat(Mu(j,:),N,1)').*Rj;
        Sigma(j,1:2,1:2)= rxmu*(X-repmat(Mu(j,:),N,1)')'/sumr;
        alpha(j)=sumr/N; % 새로운 혼합계수
    end
    if (mod(tau,10)==1)     % 그래프그리기 함수 호출
        drawgraph(X, Mu, Sigma, ceil(tau/10)+1);
    end
end

drawgraph(X, Mu, Sigma, tau); % 그래프그리기 함수 호출
figure(tau+1); plot(L); % 로그우도 그래프 그리기


%------------------------------------------------------
function [out]=gausspdf(X, mu, sigma) % 함수 정의
n=size(X, 1);       % 입력 벡터의 차원
N=size(X, 2);       % 데이터의 수
Mu=repmat(mu',1,N); % 행렬 연산을 위한 준비
% 확률밀도값 계산
out = (1/((sqrt(2*pi))^n*sqrt(det(sigma))))*exp(-diag((X-Mu)'*inv(sigma)*(X-Mu))/2);

%------------------------------------------------------
function drawgraph(X, Mu, Sigma, cnt) % 함수 정의
M=size(Mu,1);       % 성분의 수
figure(cnt);        % 데이터 그리기
plot(X(1,:), X(2,:), '*'); hold on
axis([-0.5 5.5 -0.5 3.5]); grid on
plot(Mu(:,1), Mu(:,2), 'r*'); % 평균 파라미터 그리기
for j=1:M
    sigma=reshape(Sigma(j,:,:),2,2); % 공분산에 따른 타원 그리기
    t=[-pi:0.1:pi]';
    A=sqrt(2)*[cos(t) sin(t)]*sqrtm(sigma)+repmat(Mu(j,:), size(t),1);
    plot(A(:,1), A(:,2), 'r-', 'linewidth', 2);
end

