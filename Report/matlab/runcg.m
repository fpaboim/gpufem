clc;
clear;

err     = 0.0001;
maxiter = 10;

A = [3 2; 2 6];
b = [2 -8]';
c = 0;

x0 = [0 0]';
x = CG(A, b, x0, err, maxiter);

dim = 4;
rand_range = [0,4];
R = randint(dim,dim,rand_range)
S = 10 * eye(dim,dim);
K = R' * R;
K = K + S
eig(K)

%precond stuff
diagR = diag(R);
C = eye(dim,dim);
for i = 1:dim
  C(i,i) = K(i);
end

x0(dim) = 0;
x2(dim) = 0;
x2 = randint(dim, 1, rand_range)
b = K * x2
x = CG(K, b, x0, err, maxiter)

