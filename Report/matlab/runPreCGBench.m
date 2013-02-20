clear;
clc;

% Setup Data
A   = delsq(numgrid('S',16));
dim = size(A,1);
b   = ones(dim,1);

% No preconditioner
[x0,fl0,rr0,it0,rv0] = pcg(A,b,1e-8,100);
it0

% Incomplete cholesky factorization preconditioner
L = cholinc(A, '0');
[x1,fl1,rr1,it1,rv1] = pcg(A,b,1e-8,100,L,L');
it1

% Modified incomplete cholesky factorization
opts.droptol = 0.0001;
opts.michol = 1;
L = cholinc(A, opts);
[x2,fl2,rr2,it2,rv2] = pcg(A,b,1e-8,100,L,L');
it2
rr2

% Diagonal Scaling
D = eye(dim,dim);
for i = 1:dim
  D(i,i) = A(i,i);
end
Dinv = inv(D)
B = Dinv * A;
b2 = Dinv * b;
[x3,fl3,rr3,it3,rv3] = pcg(B,b2,1e-8,100);
it3

% Plot results
figure;
semilogy(0:it0,rv0/norm(b),'b.');
hold on;
semilogy(0:it1,rv1/norm(b),'r.');
semilogy(0:it2,rv2/norm(b),'k.');
semilogy(0:it3,rv3/norm(b),'c+');
legend('No Preconditioner','IC(0)','MIC(0)', 'DIAG');
xlabel('iteration number');
ylabel('relative residual');
hold off;
