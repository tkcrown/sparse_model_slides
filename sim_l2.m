N = 30;

X = sort(rand(N,1));



XX = [X, X.^2];

XXXX = [X, X.^2, X.^3, X.^4, X.^5, X.^6, X.^7, X.^8];

beta = rand(2,1)-0.5;

Y = XX * beta;
y = normrnd(0,0.05, N, 1) + Y;




beta_hat = (XXXX' * XXXX)^(-1)*XXXX'*y;
y_hat = XXXX * beta_hat;

lambda = 0.0001;
I = eye(2);
beta_hat_reg = (XX' * XX + lambda * I)^(-1)*XX'*y;
y_hat_reg = XX * beta_hat_reg;



subplot(1,2,1)
plot(X, Y, 'c*-');
hold on;
plot(X, y, 'xr');
plot(X, y_hat, 'ob');

x = (1:100)'./100;
xxxx = [x, x.^2, x.^3, x.^4, x.^5, x.^6, x.^7, x.^8];
yy_hat = xxxx * beta_hat;
plot(x, yy_hat, '-b');

legend('E[Y|X]','y','\hat{y}');
xlabel('X')
ylabel('y')
hold off;

subplot(1,2,2)

plot(X, Y, 'c*-');
hold on;
plot(X, y, 'xr');
plot(X, y_hat_reg, 'og-');


legend('E[Y|X]','y','\hat{y}_{reg}');
xlabel('X')
ylabel('y')
hold off;
