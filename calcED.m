function ED = calcED(obj1,obj2)
%CALCED 此处显示有关此函数的摘要
%   此处显示详细说明
ED2 = 0;
for i=1:length(obj1)
    ED2 = ED2 + power(obj1(i)-obj2(i), 2);
end
ED = power(ED2, 1/2);
end

