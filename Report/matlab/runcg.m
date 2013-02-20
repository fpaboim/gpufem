clc;
clear;

err     = 0.00001;
maxiter = 1000;

A = [3 2; 2 6];
b = [2 -8]';
c = 0;

x0 = [0 0]';
x = CG(A, b, x0, err, maxiter);

dim = 5;
R = rand(dim,dim);
K = R' * R;
K = K + eye(dim,dim);

diagR = diag(R);
C = eye(dim,dim);
for i = 1:dim
  C(i,i) = K(i);
end

x0(dim) = 0;
x2(dim) = 0;
x2 = rand(dim,1);
b = K * x2;
x = CG(K, b, x0, err, maxiter);
