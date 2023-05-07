function [Wm,Wc,Wr,Hm,Hc,Hr] = clusterH(W,WB,H,basis_num);
%CLUSTER 此处显示有关此函数的摘要
%   此处显示详细说明
%basis_num = 8;
cf = zeros(basis_num,basis_num);
for i = 1:basis_num
    for j = 1:basis_num
        mid1 = W(:,i)'*WB(:,j);
        mid2 = (W(:,i)'*W(:,i))^0.5*(WB(:,j)'*WB(:,j))^0.5;
        cf(i,j) = mid1/mid2;
    end
    score(i) = max(cf(i,:));
    
end

alpha = max(score) - max(score)/(2+8);
beta = min(score) + min(score)/(2+0);
if alpha < beta
    alpha = max(score) - max(score)/(2+18);
    beta = min(score) + min(score)/(2+2);
end
Wc = W(:,score>alpha);
Wr = W(:,score<beta);
Wm = W(:,score<=alpha & score>=beta);
Hc = H(score>alpha,:);
Hr = H(score<beta,:);
Hm = H(score<=alpha & score>=beta,:);

