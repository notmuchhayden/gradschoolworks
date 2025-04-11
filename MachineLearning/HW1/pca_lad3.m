% PCA 함수 구현
function X_pca = PCA_Custom(X, n_components)
    mean_X = mean(X, 1);
    X_centered = X - mean_X;
    cov_matrix = cov(X_centered);
    [eig_vectors, eig_values] = eig(cov_matrix);
    eig_values = diag(eig_values);

    [sorted_eig_values, idx] = sort(eig_values, 'descend');
    sorted_eig_vectors = eig_vectors(:, idx);

    % 차원 선택 (예: 95% 이상 정보 보존)
    if n_components <= 1
        cumulative_variance = cumsum(sorted_eig_values) / sum(sorted_eig_values);
        n_components = find(cumulative_variance >= n_components, 1);
    end

    components = sorted_eig_vectors(:, 1:n_components);
    X_pca = X_centered * components;
end

% LDA 함수 구현
function X_lda = LDA_Custom(X, Y, n_components)
    class_labels = unique(Y);
    mean_overall = mean(X, 1);
    Sw = zeros(size(X, 2));
    Sb = zeros(size(X, 2));

    for i = 1:length(class_labels)
        class_idx = find(Y == class_labels(i));
        X_class = X(class_idx, :);
        mean_class = mean(X_class, 1);
        Sw = Sw + cov(X_class) * (size(X_class, 1) - 1);
        mean_diff = mean_class - mean_overall;
        Sb = Sb + size(X_class, 1) * (mean_diff' * mean_diff);
    end

    [eig_vectors, eig_values] = eig(pinv(Sw) * Sb);
    eig_values = diag(eig_values);
    [~, idx] = sort(eig_values, 'descend');
    eigen_vectors = eig_vectors(:, idx(1:n_components));
    X_lda = X * eigen_vectors;
end

% 데이터 로드
load('HW1_COIL20.mat');
X = X'; % 전치하여 행렬 형식 맞추기
Y = Y(:);

% PCA 실행 (2차원 축소)
X_pca = PCA_Custom(X, 2);
X_pca(:, 1) = -X_pca(:, 1); % 좌우 반전
X_pca(:, 2) = -X_pca(:, 2); % 상하 반전

% PCA 결과 시각화
figure;
subplot(1,2,1);
hold on;
for label = unique(Y)'
    scatter(X_pca(Y == label, 1), X_pca(Y == label, 2), 'DisplayName', sprintf('Class %d', label));
end
hold off;
legend('Location', 'northeast');
title('PCA 2D Projection');

% PCA (95% 정보 유지) 후 LDA 적용
X_pca_95 = PCA_Custom(X, 0.95);
X_lda = LDA_Custom(X_pca_95, Y, 2);
X_lda(:, 1) = -X_lda(:, 1); % LDA 결과 좌우 반전
X_lda(:, 2) = -X_lda(:, 2); % LDA 결과 상하 반전

% LDA 결과 시각화
subplot(1,2,2);
hold on;
for label = unique(Y)'
    scatter(X_lda(Y == label, 1), X_lda(Y == label, 2), 'DisplayName', sprintf('Class %d', label));
end
hold off;
legend('Location', 'northeast');
title('LDA 2D Projection');

