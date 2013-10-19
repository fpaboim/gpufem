function [u, niter, flag] = CG(A, b, x0, tol, maxiter)
u   = x0;         % Set u_0 to the start vector x0
r   = b - A*x0;   % Compute first residuum
p   = r;
rho = r'*r;

niter = 0;     % Init counter for number of iterations
flag  = 0;      % Init break flag
bnorm = norm(b);
if bnorm < tol  % if the norm is very close to zero, take the
                % absolute residuum instead as break condition
                % ( norm(r) > tol ), since the relative
                % residuum will not work (division by zero).
  warning(['norm(f) is very close to zero, taking absolute residuum' ...
           ' as break condition.']);
  bnorm = 1;
end

while (norm(r)/bnorm > tol)   % Test break condition
  a = A*p;
  alpha = rho/(a'*p);
  u = u + alpha*p;
  r = r - alpha*a;
  rho_new = r'*r;
  p = r + rho_new/rho * p;
  rho = rho_new;
  niter = niter + 1;
  if (niter >= maxiter)         % if max. number of iterations
    flag = 1;                   % is reached, break.
    break
  end
end
niter
