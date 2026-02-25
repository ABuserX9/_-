function interpolated_solutions = OT_Wasserstein_Interpolation2(Problem, startSolutions_decs, targetSolutions_decs, num_solutions_to_generate, initial_lambda)
% OT_Wasserstein_Interpolation2: 使用最优传输(Optimal Transport)和分批次插值策略来生成新的解。
%
%   该函数是CPI-DCMOEA算法的核心部分，用于在环境变化后生成高质量的初始种群。
%   其基本思想是：将一组解（如随机解）视为质量分布的“起点”，另一组解（如预测的UPS）视为“目标点”。
%   通过求解最优传输问题，找到一个从起点到目标点最经济的“传输方案”。
%   然后，沿着这个方案定义的“收敛路径”进行插值，生成新的解。
%   插值过程是分批次的，并且会根据生成解的质量动态调整插值区间，从而聚焦于更有希望的区域。
%
%   输入:
%       Problem                  - 问题定义 (结构体)
%       startSolutions_decs      - 作为起点的解的决策变量矩阵 (例如，随机解)
%       targetSolutions_decs     - 作为目标的解的决策变量矩阵 (例如，预测的UPS)
%       num_solutions_to_generate - 需要生成的插值解的数量
%       initial_lambda           - 初始插值参数 (在当前实现中被内部的批次插值参数覆盖)
%
%   输出:
%       interpolated_solutions   - 通过插值生成的新解的决策变量矩阵

    % --- 1. 初始化和参数准备 ---
    % 确保插值参数lambda在[0, 1]区间内 (尽管外部传入的lambda未在主循环中使用)
    lambda = max(0, min(1, initial_lambda));
    
    % 获取起点和目标点种群的大小及决策空间维度
    num_start_solutions = size(startSolutions_decs, 1);
    num_target_solutions = size(targetSolutions_decs, 1);
    num_dimensions = size(startSolutions_decs, 2);
    
    % 检查维度一致性
    if num_dimensions ~= size(targetSolutions_decs, 2)
        error('起点和目标解集必须有相同的决策变量维度。');
    end

    %% 2. 求解最优传输问题 (Optimal Transport Problem)
    % --- 2.1 计算代价矩阵 ---
    % 代价矩阵C(i,j)表示将质量从起点i移动到目标点j的成本。
    % 这里使用欧氏距离的平方作为代价函数。
    costMatrix = pdist2(startSolutions_decs, targetSolutions_decs, 'euclidean').^2;

    % --- 2.2 定义边缘分布 ---
    % 假设起点和目标点的质量是均匀分布的。
    start_marginal_dist = ones(num_start_solutions, 1) / num_start_solutions;
    target_marginal_dist = ones(num_target_solutions, 1) / num_target_solutions;

    % --- 2.3 设置线性规划问题 ---
    % 将代价矩阵展平为列向量，作为线性规划的目标函数系数。
    f = costMatrix(:);

    % 构造等式约束 Aeq * x = beq，确保传输方案满足边缘分布。
    % 传输方案gamma是一个 N x M 的矩阵，同样被展平为向量处理。
    % 约束1: gamma的所有列相加应等于起点的边缘分布p。
    Aeq1 = kron(speye(num_start_solutions), ones(1, num_target_solutions));
    beq1 = start_marginal_dist;
    % 约束2: gamma的所有行相加应等于目标点的边缘分布q。
    Aeq2 = kron(ones(1, num_start_solutions), speye(num_target_solutions));
    beq2 = target_marginal_dist;
    
    Aeq = [Aeq1; Aeq2];
    beq = [beq1; beq2];

    % 设置变量下界（传输量不能为负）
    lb = zeros(num_start_solutions * num_target_solutions, 1);
    ub = []; % 无上界

    % --- 2.4 求解线性规划 ---
    % 使用内点法求解，不显示迭代过程。
    options = optimoptions('linprog', 'Algorithm', 'interior-point', 'Display', 'none');
    [transportPlan_vector, ~, exitflag] = linprog(f, [], [], Aeq, beq, lb, ub, options);
    
    if exitflag ~= 1
        warning('最优传输问题求解失败，将使用次优解继续。');
    end
    
    % 如果求解失败，使用一个均匀分布的传输方案作为备用
    if isempty(transportPlan_vector)
        transportPlan_vector = ones(1, num_start_solutions * num_target_solutions) * (1 / (num_start_solutions * num_target_solutions));
    end

    % 将解向量重塑为 N x M 的传输方案矩阵
    transportPlan = reshape(transportPlan_vector, [num_start_solutions, num_target_solutions]);
    
    %% 3. 分批次插值与自适应区间调整
    % --- 3.1 准备采样 ---
    % 将传输方案归一化为概率分布，并计算累积分布函数(CDF)用于抽样。
    transportPlan_prob = transportPlan / sum(transportPlan(:));
    transportPlan_prob_vec = transportPlan_prob(:);
    transportPlan_cdf = cumsum(transportPlan_prob_vec);
    
    % 生成用于采样的随机数
    rand_samples = rand(num_solutions_to_generate, 1);
    
    % --- 3.2 初始化插值过程 ---
    interpolated_solutions = zeros(num_solutions_to_generate, num_dimensions);
    batch_size = num_solutions_to_generate / 10; % 将总数分为10个批次
    
    % 初始化插值参数lambda的搜索区间
    interval_start = 0;
    interval_end = 1;
    
    % --- 3.3 循环生成和评估每个批次的解 ---
    for batch = 1:10
        % 在当前区间内均匀生成一批插值参数
        lambdas_for_batch = linspace(interval_start, interval_end, batch_size);
        
        % --- 为当前批次生成插值解 ---
        for i = 1:batch_size
            current_solution_index = (batch - 1) * batch_size + i;
            
            % 根据随机数和CDF确定采样的传输对(i, j)
            sampled_pair_flat_index = find(rand_samples(current_solution_index) <= transportPlan_cdf, 1, 'first');
            [start_sol_idx, target_sol_idx] = ind2sub([num_start_solutions, num_target_solutions], sampled_pair_flat_index);
            
            % 执行线性插值
            current_lambda = lambdas_for_batch(i);
            interpolated_solutions(current_solution_index, :) = (1 - current_lambda) * startSolutions_decs(start_sol_idx, :) + current_lambda * targetSolutions_decs(target_sol_idx, :);
        end
        
        % --- 评估当前批次的解并调整插值区间 ---
        % 提取并评估当前批次生成的解
        current_batch_solutions_decs = interpolated_solutions((batch - 1) * batch_size + 1 : batch * batch_size, :);
        evaluated_batch_solutions = Problem.Evaluation(current_batch_solutions_decs, false);
        
        % 根据非支配排序和拥挤度对批次内的解进行排序
        [frontNo_batch, ~] = NDSort(evaluated_batch_solutions.objs, inf); % 注意：原始代码使用.mFs，这里遵循原始实现
        crowd_batch = CrowdingDistance(evaluated_batch_solutions.objs);
        [~, sorted_indices_in_batch] = sortrows([frontNo_batch', -crowd_batch']);
        
        % 计算排名前列的解的平均索引，以判断哪个子区间更优
        % 如果最优解主要出现在前半部分，则缩小搜索区间到左半边，反之亦然
        mean_rank_index = mean(sorted_indices_in_batch(1:max(1, batch_size-5))); % 取前 batch_size-5 个解
        
        if mean_rank_index < batch_size / 2
            interval_end = (interval_start + interval_end) / 2;
        elseif mean_rank_index > batch_size / 2
            interval_start = (interval_start + interval_end) / 2;
        end
    end
    
    %% 4. 确保生成的解在边界范围内
    interpolated_solutions = max(interpolated_solutions, Problem.lower);
    interpolated_solutions = min(interpolated_solutions, Problem.upper);
end
