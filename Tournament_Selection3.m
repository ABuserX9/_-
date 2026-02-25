function index = Tournament_Selection(varargin)

%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
    K=3;
    N=3;
    varargin    = cellfun(@(S)reshape(S,[],1),varargin,'UniformOutput',false);
    [Fit,~,Loc] = unique([varargin{:}],'rows');
    [~,rank]    = sortrows(Fit);
    [~,rank]    = sort(rank);
    Parents     = randi(length(varargin{1}),K,N);
    [~,best]    = min(rank(Loc(Parents)),[],1);
    index       = Parents(best+(0:N-1)*K);
end