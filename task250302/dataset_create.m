% 设置随机数种子，保证结果可重复
rng(0);

% 生成一个 100x10 的随机数据集
data = randn(100, 10);

% 添加一些相关性，使 PCA 更有效
for i = 2:10
    data(:, i) = data(:, i) + 0.5 * data(:, i-1);
end

% 将数据集保存到文件（可选）
save('my_data.mat', 'data');