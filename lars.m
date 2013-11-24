
N = 2;
p = 3;
X = [2, 2, -2; 0,1,0];



avg = sum(X)/N;

nm = sqrt(sum(X.^2));

C_X = X./repmat(nm, 2, 1);

residual = y - sum(y)/numel(y);

beta = zeros(p, 1);

corr = abs(C_X' * residual);

[value, indx] = max(corr);

Active_set = [indx];


for i = 1:(p-1)
    fit = C_X(:, Active_set) * beta(Active_set);
    residual = y - fit;
    
    
    
end