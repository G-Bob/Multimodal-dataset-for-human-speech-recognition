function [Wm,Wc,Wr,Hm,Hc,Hr] = clusterM(Wm,Wc,Wr,Hm,Hc,Hr)

cf = zeros(size(Wm,2), size(Wc,2));
for i = 1:size(Wm,2)
    for j = 1:size(Wc,2);
        mid1 = Wm(:,i)'*Wc(:,j);
        mid2 = (Wm(:,i)'*Wm(:,i))^0.5*(Wc(:,j)'*Wc(:,j))^0.5;
        cf(i,j) = mid1/mid2;
    end
    score(i) = max(cf(i,:));
end
alpha = max(score) - max(score)/(2+8);
beta = min(score) + min(score)/(2+0);
if alpha < beta
    alpha = max(score) - max(score)/(2+6);
    beta = min(score) + min(score)/(2+2);
end
Wc = [Wc Wm(:,score>alpha)];
Wr = [Wr Wm(:,score<beta)];
Wm = Wm(:,score<=alpha & score>=beta);
Hc = [Hc;Hm(score>alpha,:)];
Hr = [Hr;Hm(score<beta,:)];
Hm = Hm(score<=alpha & score>=beta,:);
end

