
%------------------------------- Copyright --------------------------------
% Copyright (c) 2023 BIMK Group. You are free to use the PlatEMO for
% research purposes. All publications which use this platform or any code
% in the platform should acknowledge the use of "PlatEMO" and reference "Ye
% Tian, Ran Cheng, Xingyi Zhang, and Yaochu Jin, PlatEMO: A MATLAB platform
% for evolutionary multi-objective optimization [educational forum], IEEE
% Computational Intelligence Magazine, 2017, 12(4): 73-87".
%--------------------------------------------------------------------------
function dec = Init(Problem)
%INIT 此处显示有关此函数的摘要
%   此处显示详细说明
    dec = zeros(Problem.N,Problem.D);
    U = Problem.upper;
    L=  Problem.lower;
    for i=1:Problem.N
        for j=1:Problem.D
            dec(i,j) = L(j)+rand(1,1)*(U(j)-L(j));
        end
    end
end

