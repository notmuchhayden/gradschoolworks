load data12_20                          % 데이터 불러오기
N = size(X, 1);                         % 데이터의 수
INP = 2; HID = 5; OUT =  2;             % 뉴런의 수 설정
w = rand(INP, HID)*0.4 - 0.2;           % 입력->은닉 뉴런 가중치 초기화
w0 = rand(1, HID)*0.4 - 0.2;            % 바이어스
v = rand(HID, OUT)*0.4 - 0.2;           % 은닉->출력 뉴런 가중치 초기화
v0 = rand(1, OUT)*0.4 - 0.2;
eta = 0.001;                            % 학습률 설정
Mstep = 5000; Elimit=0.05;              % 박복횟수 설정

for j = 2:Mstep                         % 학습 반복 시작
    for i = 1:N                         % 각 데이터에 대한 반복 시작
        x = X(i, : ); t = T(i, : );     % 입력과 목표 출력 데이터 선택
        uh = x*w + w0;                  % 은닉 뉴런의 가중합 계산
        z = tanh(uh);                   % 은닉 뉴런의 출력 계산
        uo = z*v + v0;                  % 출력 뉴런의 가중합 계산
        y = tanh(uo);                   % 출력 뉴런의 출력 계산
        e = y - t;                      % 신경망 출력과 목표 출력의 차 계산
        E(i, 1) = e*e';                 % 제곱오차 계산
        delta_v = d_tanh(y).*e;         % 학습을 위한 델타값 계산 (출력뉴런)
        delta_w = d_tanh(z).*(delta_v*v');% 학습을 위한 델타값 계산 (은닉뉴런)
        v = v - eta*(z'*delta_v);       % 출력 뉴런의 가중치 수정
        v0 = v0 - eta*(1*delta_v);      % 출력 뉴런의 바이어스 가중치 수정
        w = w - eta*(x'*delta_w);       % 은닉 뉴런의 가중치 수정
        w0 = w0 - eta*(1*delta_w);      % 은닉 뉴런의 바이어스 가중치 수정
    end
    [serr, cerr] = MLPtest(X, T, w, w0, v, v0); % 학습 오차 계산 함수 호출
    fprintf(1, '%d %7.3f %7.3f\n', j, serr, cerr); % 오차 변화 출력
    Serr(j-1, : ) = serr; Cerr(j-1, : ) = cerr; % 오차 변화 저장
    if (serr<Elimit) break; end
end

save MLPweight INP HID OUT v v0 w w0    % 가중치와 신경망 구조 저장

% 입력 벡터를 받아서 하이퍼탄젠트 함수의 1차 도함수의 값을 반환
function [out]=d_tanh(a)
    out = (1-a).*(1+a);
end

% 데이터 집합과 가중치를 받아 평균제곱오차와 분류오차를 계산
function [SEtst, CEtst] = MLPtest(Xtst, Ttst, w, w0, v, v0)
    N = size(Xtst, 1);                  % 데이터의 수
    for i = 1:N                         % 각 데이터에 대한 인식 시작
        x = Xtst(i, : ); t = Ttst(i, : ); % 입/출력 데이터 설정
        uh = x*w + w0;                  % 은닉 뉴런의 가중합 계산
        z = tanh(uh);                   % 은닉 뉴런의 출력 계산
        uo = z*v + v0;                  % 출력 뉴런의 가중합 계산
        y = tanh(uo);                   % 출력 뉴런의 출력 계산
        e = y - t;                      % 목표 출력과의 차 계산
        E(i, 1) = e*e';                 % 제곱오차 계산
        if y(1) > y(2)
            Ytst(i, :) = [1, -1];       % 최종 인식 결과 판단
        else
            Ytst(i, :) = [-1, 1];
        end
    end
    SEtst = sum(E.^2)/N;                % 평균제곱오차 계산
    diffTY = sum(abs(Ttst - Ytst))/2;
    CEtst = diffTY(1)/N;                % 분류오차 계산
end

