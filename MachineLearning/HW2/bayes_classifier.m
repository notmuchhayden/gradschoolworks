%베이즈 분류기를 이용하여 데이터를 분류하고 학습 오차를 출력

load dataCh4_7 						%데이터 불러오기

K = 3; 								%클래스의 수

M = [mean(X1); mean(X2); mean(X3)] 	%클래스별 표본평균 계산

S(:,:,1) = cov(X1); 				%클래스별 표본 공분산 계산
S(:,:,2) = cov(X2);
S(:,:,3) = cov(X3);

smean = (cov(X1)+cov(X2)+cov(X3))/3 %클래스별 표본공분산의 평균
Dtrain = [X1;X2;X3]; 				%학습 데이터 구성

Etrain = zeros(3, 1); 				%오분류 데이터의 수를 셈
N = size(X1,1); 					%각 클래스별 데이터의 수
for k = 1 : K						%각 클래스별로 분류 시작
	X = Dtrain((k - 1)*100+1:k*100,:);
	for i = 1 : N 					%각 데이터에 대해 분류 시작
		for j = 1 : K 				%세 개의 판별함수의 값 계산
			%단위 공분산행렬을 가정한 경우의 판별함수
			d1(j,1) = (X(i,:) - M(j,:)) * (X(i,:) - M(j,:))';
			%모든 클래스에 동일한 공분산행렬을 가정한 경우의 판별함수
			d2(j,1) = (X(i,:) - M(j,:))*inv(smean)*(X(i,:)-M(j,:))';
			%일반적인 공분산행렬을 가정한 경우의 판별함수
			d3(j,1) = (X(i,:) - M(j,:))*inv(reshape(S(:,:,j),2,2))*(X(i,:)-M(j,:))';
		end
		
		[min1v, min1i]=min(d1); 	% 각 판별함수 값에 따라 분류
		if (min1i ~= k)
			Etrain(1, 1) = Etrain(1, 1) + 1;
		end
		
		[min2v, min2i]=min(d2);		% 원래 클래스와 다르면 오류증가
		if (min2i ~= k)
			Etrain(2, 1) = Etrain(2, 1) + 1;
		end
		
		[min3v, min3i]=min(d3);		% 원래 클래스와 다르면 오류증가
		if (min3i ~= k)
			Etrain(3, 1) = Etrain(3, 1) + 1;
		end
	end
end
Error_rate = Etrain/(N*K)