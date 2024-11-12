function [out] = Positional_Encoding(X)
%POSITIONAL_ENCODING 此处显示有关此函数的摘要
%   此处显示详细说明

length = size(X,1);

for pos = 0:length-1
    if mod(pos,2)==1
        X(pos+1,1) = X(pos+1,1) + cos(length/(10000^((pos-1)/length)));
    else
        X(pos+1,1) = X(pos+1,1) + sin(length/(10000^(pos/length)));
    end
end
out = X;
end

