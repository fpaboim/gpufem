clc;
clear;

err     = 0.00001;
maxiter = 1000;

R = 'L'; % Other possible shapes include S,N,C,D,A,H,B
% Generate and display the grid.
n = 32;
G = numgrid(R,n);
%spy(G)
title('A finite difference grid')
% Show a smaller version as sample.
g = numgrid(R,12)

% Discrete Laplacian
D = delsq(G);
spy(D)
title('The 5-point Laplacian')
% Number of interior points
N = sum(G(:)>0)

% Dirichlet boundary value problem
rhs = ones(N,1);
if (R == 'N') % For nested dissection, turn off minimum degree ordering.
    spparms('autommd',0)
    u = D\rhs;
    spparms('autommd',1)
else
    u = D\rhs; % This is used for R=='L' as in this example
end

% Solution
U = G;
U(G>0) = full(u(G(G>0)));
clabel(contour(U));
prism
axis square ij

colormap((cool+1)/2);
mesh(U)
axis([0 n 0 n 0 max(max(U))])
axis square ij
