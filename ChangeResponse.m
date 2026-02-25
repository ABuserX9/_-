function [newMainPopulation, newAuxiliaryPopulation, predictionModel] = ChangeResponse(Problem, Population, auxiliaryPopulation, constrainedArchive, unconstrainedArchive, historicalMainPopulations, historicalAuxiliaryPopulations, predictionModel)
% ChangeResponse_mutate: 当检测到环境变化时，执行CPI-DCMOEA的核心响应策略。
%   该函数是论文 "A Dynamic Constrained Multiobjective Evolutionary Algorithm based on Convergence Path Interpolation" 
%   中描述的“收敛路径插值”策略的MATLAB近似实现。
%
%   主要流程如下:
%   1.  如果历史环境数据充足(>=2)，则执行预测和插值：
%       a.  首先，使用一个基于Transformer的自编码器模型，根据历史的无约束种群(UPS)来预测新环境的UPS。
%       b.  然后，调用最优传输(Optimal Transport)插值函数，在随机解和预测出的UPS之间构建“收敛路径”，并沿路径生成新的高质量解。
%       c.  最后，将插值生成的解、上一代精英解和随机解组合，构成新环境的主种群(NCP)。
%   2.  如果历史数据不足，则采用一种保守的精英保留和随机生成策略来初始化新种群。
%
%   输入:
%       Problem                      - 问题定义 (结构体)
%       Population                   - 当前主种群 (有约束)
%       auxiliaryPopulation          - 当前辅助种群 (无约束, 即Upop)
%       constrainedArchive           - 上一环境的有约束存档 (A)
%       unconstrainedArchive         - 上一环境的无约束存档 (UA)
%       historicalMainPopulations    - 所有历史环境的主种群集合
%       historicalAuxiliaryPopulations - 所有历史环境的辅助种群集合
%       predictionModel              - 用于预测UPS的Transformer自编码器模型
%
%   输出:
%       newMainPopulation            - 为新环境生成的主种群 (NCP)
%       newAuxiliaryPopulation       - 为新环境生成的辅助种群 (NUP)
%       predictionModel              - 经过训练更新后的预测模型

    % --- 初始化参数 ---
    predictionPopulationSize = Problem.N * 0.5; % 预测器使用的种群大小
    lambda = 0.001; % (未使用) lambda参数
    upperBounds = Problem.upper; % 决策变量上界
    lowerBounds = Problem.lower; % 决策变量下界
    
    % 为历史不足时的回退策略设置精英保留比例
    eliteRatioConstrained = 0.1; 
    eliteRatioUnconstrained = 0.0;
    
    % --- 对上一代存档按拥挤度距离进行排序，以备在历史数据不足时选用精英解 ---
    [~, constrainedArchiveSortPermutation] = sortrows([CrowdingDistance(constrainedArchive.objs)']);
    [~, unconstrainedArchiveSortPermutation] = sortrows([CrowdingDistance(unconstrainedArchive.objs)']);
    [~, mainPopSortPermutation] = sortrows([CrowdingDistance(Population.objs)']);
    
%------------------------------- 预测与插值核心逻辑 ------------------------------------
% 检查是否有足够的历史环境数据来执行预测（至少需要两个过去的环境）
if size(historicalMainPopulations, 1) >= 2


    %% 1. 准备训练数据
    % 提取最近的两个环境(t-1 和 t)的种群数据用于模型训练和预测
    ps_t_minus_1_pop = historicalMainPopulations(end-1, :);
    ups_t_minus_1_pop = historicalAuxiliaryPopulations(end-1, :);
    ps_t_pop = historicalMainPopulations(end, :);
    ups_t_pop = historicalAuxiliaryPopulations(end, :);

    % --- 从 t-1 环境的种群中选择并排序决策变量 ---
    [frontNo_ps_t_minus_1, ~] = NDSort(ps_t_minus_1_pop.objs, inf);
    crowd_ps_t_minus_1 = CrowdingDistance(ps_t_minus_1_pop.objs);
    [~, sortIdx_ps_t_minus_1] = sortrows([frontNo_ps_t_minus_1', -crowd_ps_t_minus_1']);
    ps_t_minus_1_decs = ps_t_minus_1_pop(sortIdx_ps_t_minus_1(1:predictionPopulationSize)).decs;
    ps_t_minus_1_decs = permute_matrix(ps_t_minus_1_decs); % 保证顺序一致性

    [frontNo_ups_t_minus_1, ~] = NDSort(ups_t_minus_1_pop.objs, inf);
    crowd_ups_t_minus_1 = CrowdingDistance(ups_t_minus_1_pop.objs);
    [~, sortIdx_ups_t_minus_1] = sortrows([frontNo_ups_t_minus_1', -crowd_ups_t_minus_1']);
    ups_t_minus_1_decs = ups_t_minus_1_pop(sortIdx_ups_t_minus_1(1:predictionPopulationSize)).decs;
    ups_t_minus_1_decs = permute_matrix(ups_t_minus_1_decs);

    % --- 从 t 环境的种群中选择并排序决策变量 ---
    [frontNo_ps_t, ~] = NDSort(ps_t_pop.objs, inf);
    crowd_ps_t = CrowdingDistance(ps_t_pop.objs);
    [~, sortIdx_ps_t] = sortrows([frontNo_ps_t', -crowd_ps_t']);
    ps_t_decs = ps_t_pop(sortIdx_ps_t(1:predictionPopulationSize)).decs;
    ps_t_decs = permute_matrix(ps_t_decs);
    
    % 仅用于后续精英保留的排序索引
    [~, sortIdx_ps_t_for_reservation] = sortrows([-crowd_ps_t']);

    [frontNo_ups_t, ~] = NDSort(ups_t_pop.objs, inf);
    crowd_ups_t = CrowdingDistance(ups_t_pop.objs);
    [~, sortIdx_ups_t] = sortrows([frontNo_ups_t', -crowd_ups_t']);
    ups_t_decs = ups_t_pop(sortIdx_ups_t(1:predictionPopulationSize)).decs;
    ups_t_decs = permute_matrix(ups_t_decs);
    
    %% 2. 预测新环境的无约束最优集 (UPS Prediction)
    % 定义模型训练参数
    numEpochs = 1;      % 训练周期
    learningRate = 1e-4; % 学习率
    
    % 增量式训练模型：用 t-1 时刻的UPS去预测 t 时刻的UPS
    predictionModel.train(ups_t_minus_1_decs, ups_t_decs, numEpochs, learningRate);

    % 使用训练好的模型，基于 t 时刻的UPS，预测 t+1 时刻的UPS
    predictedUPS_decs = predictionModel.predict(ups_t_decs);
    predictedUPS_decs = reshape(predictedUPS_decs, predictionPopulationSize, Problem.D);

    % 检查并修正超出边界的决策变量值
    outOfBounds = predictedUPS_decs > upperBounds | predictedUPS_decs < lowerBounds;
    randomValues = lowerBounds + rand(size(predictedUPS_decs)) .* (upperBounds - lowerBounds);
    predictedUPS_decs(outOfBounds) = randomValues(outOfBounds);
    predictedUPS_decs = permute_matrix(predictedUPS_decs);
    
    % 评估预测出的解
    predictedUPS = Problem.Evaluation(predictedUPS_decs, false);

    %% 3. 构造新一代种群
    % --- 3.1 构造新的辅助种群 (NUP) ---
    % 策略：预测的UPS + 随机解 + 上一代精英解
    retainedUnconstrainedSolutions = Population(sortIdx_ups_t(1:round(0.25 * Problem.N)));
    randomSolutionsForNUP = lhs_sample(lowerBounds, upperBounds, Problem.N * 0.25);
    randomSolutionsForNUP = Problem.Evaluation(randomSolutionsForNUP, false);
    
    newAuxiliaryPopulation = [predictedUPS, randomSolutionsForNUP, retainedUnconstrainedSolutions];
    newAuxiliaryPopulation = Problem.Evaluation(newAuxiliaryPopulation.decs, false);

    % --- 3.2 构造新的主种群 (NCP) ---
    % 策略：收敛路径插值生成的解 + 上一代精英解 + 随机解
    
    % 步骤 a: 生成用于插值的随机解（作为收敛路径的起点）
    randomSolutionsForInterpolation = lhs_sample(lowerBounds, upperBounds, Problem.N * 0.5);
    randomSolutionsForInterpolation = Problem.Evaluation(randomSolutionsForInterpolation, false);

    % 步骤 b: 执行收敛路径插值
    interpolatedSolutions_decs = OT_Wasserstein_Interpolation2(Problem, predictedUPS_decs, randomSolutionsForInterpolation.decs, Problem.N * 0.5, 0.5);

    % 步骤 c: 组合新种群
    retainedConstrainedSolutions = Population(sortIdx_ps_t_for_reservation(1:round(0.25 * Problem.N)));
    randomSolutionsForNCP = lhs_sample(lowerBounds, upperBounds, Problem.N * 0.25);
    
    % 将插值解、保留的精英解、随机解合并，并进行评估
    newMainPopulation_decs = [interpolatedSolutions_decs; retainedConstrainedSolutions.decs; randomSolutionsForNCP];
    newMainPopulation = Problem.Evaluation(newMainPopulation_decs, false);

else
    %% 历史数据不足时的回退处理逻辑 (环境变化次数 < 2)
    retainedConstrainedSolutionsFromArchive = SOLUTION.empty;
    retainedUnconstrainedSolutionsFromArchive = SOLUTION.empty;

    % 保留有约束存档中的精英解
    if eliteRatioConstrained > 0 && length(constrainedArchive) > 0
        numToRetain = min(length(constrainedArchive), Problem.N * eliteRatioConstrained);
        selectedSolutions = constrainedArchive(constrainedArchiveSortPermutation(1:numToRetain));
        retainedConstrainedSolutionsFromArchive = Problem.Evaluation(selectedSolutions.decs, false);
    end

    % 保留无约束存档中的精英解
    if eliteRatioUnconstrained > 0 && length(unconstrainedArchive) > 0
        numToRetain = min(length(unconstrainedArchive), Problem.N * eliteRatioUnconstrained);
        selectedSolutions = unconstrainedArchive(unconstrainedArchiveSortPermutation(1:numToRetain));
        retainedUnconstrainedSolutionsFromArchive = Problem.Evaluation(selectedSolutions.decs, false);
    end
    
    % 生成随机解以填满种群
    numRandom = Problem.N - (length(retainedConstrainedSolutionsFromArchive) + length(retainedUnconstrainedSolutionsFromArchive));
    randomSolutions = SOLUTION.empty;
    if numRandom > 0
        random_decs = lhs_sample(lowerBounds, upperBounds, numRandom);
        randomSolutions = Problem.Evaluation(random_decs, false);
    end
    
    % 组合成初始种群
    initialPopulation = [randomSolutions, retainedUnconstrainedSolutionsFromArchive, retainedConstrainedSolutionsFromArchive];
    initialPopulation = Problem.calModifiedFunction(initialPopulation.decs, initialPopulation.objs, initialPopulation.cons);
    
    % 在此情况下，主种群和辅助种群相同
    newMainPopulation = initialPopulation;
    newAuxiliaryPopulation = initialPopulation;
end
end

function permutedM = permute_matrix(M)
    %permute_matrix: 对矩阵行进行排序以保证处理顺序的一致性。
    %   排序是基于决策变量的前几维进行的，这有助于模型学习。
    [n, d] = size(M);
    num_dims = min(10, d); % 只考虑前10个维度进行排序
    [~, perm_idx] = sortrows(M(:, 1:num_dims));
    permutedM = M(perm_idx, :);
end

function sample = lhs_sample(xmin, xmax, n)
    %lhs_sample: 在给定的边界内生成拉丁超立方采样(LHS)。
    %   LHS是一种分层采样方法，能以较少的样本点均匀地覆盖整个参数空间。
    d = length(xmin); % 维度
    lhs_unit = lhsdesign(n, d); % 在单位超立方体 [0,1]^d 中生成LHS
    % 将单位LHS样本缩放到实际的决策空间边界 [xmin, xmax]
    sample = xmin + lhs_unit .* (xmax - xmin);
end

