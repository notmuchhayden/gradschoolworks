% K-means 군집화 알고리즘 (K=3)
% 학습 데이터 생성 -----------------------
X=[randn(50,2);
	randn(50,2)+repmat([3 0],50,1); randn(50,2)+repmat([0 3],50,1)];
save data7_8 X;
% 대표벡터의 초기화 ----------------------
figure(1);
plot(X(:,1),X(:,2),'*'); hold on;
N=size(X,1); K=3; m=zeros(K,2);
Xlabel=zeros(N,1); i=1;
% 단계 1 -- K개의 대표벡터를 선택
while(i<=K)
	t=floor(rand*N);	% 랜덤하게 데이터 선택
	if ((X(t,:)~=m(1,:)) & (X(t,:)~=m(2,:)) & (X(t,:)~=m(3,:)))
		m(i,:)=X(t,:);	% 선택된 데이터를 대표벡터로 둠
		plot(m(i,1), m(i,2), 'ro'); % 대표벡터를 그래프에 표시
		i=i+1;
	end
end
% K-means 반복 알고리즘의 시작 ---------------
cmode=['gd'; 'b*'; 'mo']; % 클러스터 별로 표시 기호 설정
for iteration=1:10 	% 단계 4 (단계 2와 3을 반복)
	figure(iteration+1); hold on;
% 단계 2 -- 각 데이터를 가까운 클러스터에 할당
	for i=1:N
		for j=1:K
			d(j)=(X(i,:)-m(j,:))*(X(i,:)-m(j,:))';
		end
		[minv, Xlabel(i)]=min(d);
		plot(X(i,1),X(i,2),cmode(Xlabel(i),:)); % 할당된 클러스터를 표시
	end
% 단계 3 -- 대표벡터를 다시 계산
	oldm=m;
	for i=1:K
		I=find(Xlabel==i);
		m(i,:)=mean(X(I,:));
	end
	for i=1:K
		plot(m(i,1), m(i,2), 'ks'); % 수정된 대표벡터를 표시
	end
	if sum(sum(sqrt((oldm-m).^2)))<10^(-3) break; end % 반복 완료 조건 검사
end % 단계 4 (단계 2와 3을 반복)