

x1 = rand(100,1);
x2 = rand(100,1);
 
x3 = rand(1) * x1 + rand(1) * x2 + normrnd(0,0.1);

plot3(x1, x2, x3, 'x')


axis([-0.2,1.2,-0.2,1.2,0,1.5]);

grid on

X = [x1,x2,x3];

[u,d,v] = svd(X, 'econ');
