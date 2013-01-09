clear;
clc;

% tests if diagonal squared matrix is invertible
A = eye(5);
for i = 1:5
  A(i,i)  = i * i;
  yvec(i) = i * i;
end
A(1,2) = 2;
A(2,1) = 2;
A(1,1) = 2;
chol(A);
x = inv(A) * yvec'

% tests if banded matrix is invertible
dim = 32;
B = eye(dim);
for i = 1:dim
  for j = 1:dim
    B(i,j) = 1;
  end
end
for i = 1:dim
  B(i,i) = 2;
  zvec(i) = 1;
end
chol(B);
x2 = inv(B) * zvec'

% test differentiation matrix as positive definite
dim = 4;
C = eye(dim);
C = C * 2;
wvec(dim) = 1;
for i = 1:(dim-1)
  C(i,i+1) = -1;
  C(i+1,i) = -1;
  wvec(i) = 1;
end
chol(C);
inv(C);
x3 = inv(C) * wvec'
