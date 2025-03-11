% 从文件加载数据集（如果已保存）
% load('my_data.mat');

% 计算 PCA
[coeff, score, latent] = pca(data);

% coeff：主成分系数矩阵（特征向量）
% score：主成分得分（降维后的数据）
% latent：主成分的方差（特征值）

% 可视化方差解释率
explained_variance = 100 * latent / sum(latent);
figure;
plot(cumsum(explained_variance), '-o');
xlabel('主成分数量');
ylabel('累计方差解释率 (%)');
title('主成分分析 - 方差解释率');

% 选择主成分数量（例如，保留 95% 的方差）
cumulative_variance = cumsum(explained_variance);
num_components = find(cumulative_variance >= 95, 1, 'first');

% 使用选定的主成分进行降维
reduced_data = score(:, 1:num_components);

% 显示原始数据和降维后数据的维度
disp(['原始数据维度：', num2str(size(data))]);
disp(['降维后数据维度：', num2str(size(reduced_data))]);