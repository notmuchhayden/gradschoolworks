load dataCh4_7 % 학습 데이터 로드
X = [X1;X2;X3];
Etrain=0;
N = size(X, 1);
for i=1:N
	x = X(i, :); 	% 각 데이터에 대해 분류시작
	for j=1:N		% 모든 데이터와의 거리 계산
		d(j,1)=norm(x-X(j,:));
	end
	[sx,si]=sort(d); % 거리순으로 정렬
	K=5;			% K=5 로 정함
	c=zeros(3,1);
	for j=1:K		% 이웃한 K개 데이터의 라벨을 점검하여 투표수행
		if (si(j) <= 100)
			c(1) = c(1) + 1;
		end
		
		if (si(j) > 200)
			c(3) = c(3) + 1;
		end
		
		if ((si(j) > 100) & (si(j) <= 200))
			c(2) = c(2) + 1;
		end
	end
	[maxv, maxi] = max(c);	% 최대 투표수를 받은 클래스로 할당
	if (maxi ~= (floor((i - 1)/100) + 1)	%원래 클래스 라벨과 다르면 오류데이터의 개수를 증가
		Etrain(1, 1) = Etrain(1, 1) + 1;	% 오류데이터의 개수를 증가
	end
end
Error_rate = Etrain/N 	% 오분류율 출력
	
