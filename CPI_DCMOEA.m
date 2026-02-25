classdef CPI_DCMOEA < ALGORITHM
% <multi> <real/integer/label/binary/permutation> <constrained/dynamic> <dynamic>
% CPI_DCMOEA: 一种基于收敛路径插值的动态约束多目标进化算法。
% 该算法在环境变化时，通过预测无约束帕累托最优集(UPS)，并沿随机解到UPS的收敛路径进行插值，
% 来生成高质量的初始种群。

    methods
        function main(Algorithm, Problem)
            %% 1. 初始化阶段
            % -------------------------------------------------------------------------
            totalGenerations = 0; % 初始化总迭代次数计数器

            % --- 生成并评估初始种群 ---
            initialDecisionVariables = Init(Problem); % 生成初始种群的决策变量
            Population = Problem.Evaluation(initialDecisionVariables, true); % 计算初始种群的目标值和约束违反度

            % --- 初始化主种群(CP)和辅助种群(UP) ---
            % 'Population' 是主种群，处理有约束问题
            % 'auxiliaryPopulation' 是辅助种群，在无约束空间进行探索
            auxiliaryPopulation = Population;

            % --- 初始化存档(Archive) ---
            % constrainedArchive: 存储有约束环境下的非支配解 (PS的近似)
            % unconstrainedArchive: 存储无约束环境下的非支配解 (UPS的近似)
            constrainedArchive = SOLUTION.empty;
            unconstrainedArchive = SOLUTION.empty;
            constrainedArchive = NonDominatedSelection(Problem, Population, constrainedArchive);
            unconstrainedArchive = UnconstrainedNonDominatedSelection(Problem, Population, unconstrainedArchive);

            % --- 初始化历史数据存储变量 ---
            historicalMainPopulations = [];	
            historicalAuxiliaryPopulations = [];	

            % --- 初始化性能指标记录变量 ---
            igdsPerEnvironment = [];    % 存储每个环境结束时的IGD值
            igdsPerGeneration = [];     % 存储每一代的IGD值
            currentIGD = 0;             % 当前代的IGD值
            hvsPerEnvironment = [];     % 存储每个环境结束时的HV值
            currentHV = 0;              % 当前代的HV值
            allObjectivesHistory = zeros(0, Problem.M); % 记录所有种群的目标值历史

            % --- 初始化环境和迭代计数器 ---
            environmentIndex = 1;       % 当前环境的索引
            iterationsInCurrentEnv = 1; % 在当前环境中的迭代次数

            warning('off'); % 关闭不必要的警告信息

            % --- 获取真实帕累托前沿用于性能评估 ---
            trueParetoFronts = Problem.GetOptimum();

            % --- 初始化CPI预测模型 ---
            % 定义Transformer自编码器模型的维度和参数
            inputDim = Problem.D;       % 输入维度 (决策变量数量)
            featureDim = 10;            % 特征维度 (隐藏层)
            outputDim = Problem.D;      % 输出维度
            lambda = 0.1;               % 梯度反转层的lambda参数
            % 创建Transformer自编码器包装类的实例
            transformerAutoencoderModel = TransformerAutoencoderWrapper(inputDim, featureDim, outputDim, lambda);

            %% 2. 优化主循环
            % -------------------------------------------------------------------------
            while Algorithm.NotTerminated(Population)
                % --- 初始化子代种群 ---
                auxiliaryOffspring = SOLUTION.empty; % 存储由辅助种群生成的子代
                mainOffspring = SOLUTION.empty;      % 存储由主种群生成的子代

                %% 2.1 环境变更检测与响应
                if Changed(Problem, Population)
                    % --- 记录上一个环境结束时的状态 ---
                    allObjectivesHistory = [allObjectivesHistory, Population.objs];
                    iterationsInCurrentEnv = 1; % 重置环境内迭代计数

                    % 存储上一个环境最终代的性能指标
                    igdsPerEnvironment(end+1) = currentIGD;
                    hvsPerEnvironment(end+1) = currentHV;

                    % 存储历史种群，为预测提供数据
                    historicalMainPopulations = [historicalMainPopulations; Population];
                    historicalAuxiliaryPopulations = [historicalAuxiliaryPopulations; auxiliaryPopulation];

                    'Change detected' % 在控制台输出变更信息

                    % --- 执行动态响应策略 ---
                    % 该函数封装了CPI-DCMOEA的核心：UPS预测、收敛路径构建和插值
                    [Population, auxiliaryPopulation, transformerAutoencoderModel] = ChangeResponse(Problem, Population, auxiliaryPopulation, constrainedArchive, unconstrainedArchive, historicalMainPopulations, historicalAuxiliaryPopulations, transformerAutoencoderModel);

                    environmentIndex = environmentIndex + 1; % 更新环境索引
                end

                %% 2.2 进化操作：生成子代
                auxiliaryParentPool = auxiliaryPopulation; % 创建辅助种群的副本作为父代选择池
                mainPopFitness = CalFitness(Population.objs, Population.cons);       % 计算主种群的适应度
                auxPopFitness  = CalFitness(auxiliaryPopulation.objs); % 计算辅助种群的适应度

                % offspringSplitRatio: 控制由主种群和辅助种群生成后代的固定比例
                offspringSplitRatio = 0.5;

                for i = 1:Problem.N
                    if length(auxiliaryOffspring) + length(mainOffspring) >= Problem.N
                        break; 
                    end

                    % 通过锦标赛选择父代
                    mainParentIndices = Tournament_Selection3(mainPopFitness);
                    auxParentIndices  = Tournament_Selection3(auxPopFitness);

                    % 根据固定比例，交替使用GA和DE生成子代
                    if length(mainOffspring) < Problem.N * offspringSplitRatio
                        % 策略A: 由主种群(CP)通过遗传算法(GA)生成子代，侧重开发
                        newOffspring = OperatorGAhalf(Problem, [Population(mainParentIndices(1)), Population(mainParentIndices(2))], {1, 5, 1, 40});
                        mainOffspring = [mainOffspring, newOffspring];
                    else
                        % 策略B: 由辅助种群(UP)通过差分进化(DE)生成子代，侧重探索
                        newOffspring = OperatorDE(Problem, auxiliaryParentPool(auxParentIndices(1)), auxiliaryParentPool(auxParentIndices(2)), auxiliaryParentPool(auxParentIndices(3)), {1, 0.5, 1, 20});
                        auxiliaryOffspring = [auxiliaryOffspring, newOffspring];
                    end
                end

                %% 2.3 环境选择
                % 合并父代和子代，并从中为两个种群分别选择下一代
                [Population, ~] = EnvironmentalSelection([Population, auxiliaryOffspring, mainOffspring], Problem.N, true); % 有约束环境选择
                [auxiliaryPopulation, ~] = EnvironmentalSelection([auxiliaryPopulation, auxiliaryOffspring, mainOffspring], Problem.N, false); % 无约束环境选择

                Population = Problem.calModifiedFunction(Population.decs, Population.objs, Population.cons);

                %% 2.4 更新存档
                constrainedArchive = NonDominatedSelection(Problem, Population, SOLUTION.empty);
                unconstrainedArchive = UnconstrainedNonDominatedSelection(Problem, auxiliaryPopulation, SOLUTION.empty);

                %% 2.5 计算并记录当前代的性能指标
                currentIGD = calIGD(Population, Problem, trueParetoFronts);
                currentHV = calHV2(Population, Problem, trueParetoFronts);
                igdsPerGeneration(end+1) = currentIGD;

                totalGenerations = totalGenerations + 1;
                iterationsInCurrentEnv = iterationsInCurrentEnv + 1;

                %% 3. 优化结束，保存结果
                % -------------------------------------------------------------------------
                if Problem.FE >= Problem.maxFE
                    % --- 记录并计算最终性能指标 ---
                    igdsPerGeneration(end+1) = currentIGD;
                    igdsPerEnvironment(end+1) = currentIGD;
                    hvsPerEnvironment(end+1) = currentHV;
                    MIGD = sum(igdsPerEnvironment) / length(igdsPerEnvironment)
                    MHV = sum(hvsPerEnvironment) / length(hvsPerEnvironment)

                    allObjectivesHistory = [allObjectivesHistory, Population.objs];

                    % --- 将关键性能指标(MIGD, MHV)写入CSV文件 ---
                    fileName = ['CPI_DCMOEA_', Problem.name, '.csv'];
                    fileID = fopen(fileName, 'A');
                    try
                        fprintf(fileID, '%d,', Problem.taut); 
                        fprintf(fileID, '%d,', Problem.nt); 
                        fprintf(fileID, '%s,', MIGD); 
                        fprintf(fileID, '%s,', MHV); 
                        fprintf(fileID, '\n');
                    finally
                       fclose(fileID);
                    end

                    % --- 将每一代的IGD值写入CSV文件 ---
                    fileName = ['CPI_DCMOEA_allMIGD_', Problem.name, '.csv'];
                    fileID = fopen(fileName, 'A');
                    try
                        fprintf(fileID, '%s,', igdsPerGeneration); 
                        fprintf(fileID, '\n');
                    finally
                       fclose(fileID);
                    end

                    % --- 将每个环境的IGD值写入CSV文件 ---
                    fileName = ['CPI_DCMOEA_MIGD_', Problem.name, '.csv'];
                    fileID = fopen(fileName, 'A');
                    try
                        fprintf(fileID, '%s,', igdsPerEnvironment); 
                        fprintf(fileID, '\n');
                    finally
                       fclose(fileID);
                    end

                    % --- 保存所有历史和最终的帕累托前沿解 ---
                    Population = [historicalMainPopulations; Population]; % 合并所有解

                    % 将所有解的目标值写入CSV文件
                    fileName = ['CPI_DCMOEA_PFs_', Problem.name, '.csv'];
                    fileID = fopen(fileName, 'A');
                    try
                        fprintf(fileID, '%.10f,', Population.objs);
                        for i = 1:size(Population, 1)
                            tmpPop = Population(i,:);
                            fprintf(fileID, '%.10f,', tmpPop.objs);
                        end
                        fprintf(fileID, '\n');
                    finally
                       fclose(fileID);
                    end

                    % 将最终的所有解的目标值保存为.mat文件
                    [~, rank] = sort(Population.adds(zeros(length(Population), 1)));
                    Population = Population(rank);
                    PopObjs = Population.objs;
                    matFileName = ['./PFs/CPI_DCMOEA_PFs_', Problem.name, '.mat'];
                    save(matFileName, 'PopObjs');
                end
            end
        end
    end
end

