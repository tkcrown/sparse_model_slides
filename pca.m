

x = rand(100,1);
y = rand(100,1);

z = rand(1) * x + rand(1) * y;

plot3(x, y, z, 'x')


axis([-0.2,1.2,-0.2,1.2,0,1.5]);

grid on
