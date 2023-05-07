function [W,H] = NMF_GD(X, r, max_iter, learning_rate)
sizeD = size(X); % Get V-dimension matrix as input
n = sizeD(1);
m = sizeD(2);
W = abs(rand(n,r));
H = abs(rand(r,m));

for iterate = 1:max_iter
    WD = W'*X;
    WWH = W'*W*H + learning_rate;
    H = H.*WD./WWH;
    DH = X*H';
    WHH = W*H*H' + learning_rate;
    W = W.*DH./WHH;
end
%d = W*H;

