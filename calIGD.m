function IGD = calIGD(Population,Problem,optimums)
%CALIGD 计算IGD
%   此处显示详细说明
    PopObj = Population.objs;
    t = max(0,ceil((Problem.FE/Problem.N-Problem.preEvolution)/Problem.taut));
    optimum = optimums(t+1).PF;
    if size(PopObj,2) ~= size(optimum,2)
        length(optimum);
        IGD = nan;
    else
        IGD = mean(min(pdist2(optimum,PopObj),[],2))

    end
end

