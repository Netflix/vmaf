% V = VAR2(MTX,MEAN)
%
% Sample variance of a matrix.
%  Passing MEAN (optional) makes the calculation faster.

function res = var2(mtx, mn)

if (exist('mn') ~= 1)
  mn =  mean2(mtx);
end

if (isreal(mtx))
  res = sum(sum(abs(mtx-mn).^2)) / max((prod(size(mtx)) - 1),1);
else
  res = sum(sum(real(mtx-mn).^2)) + i*sum(sum(imag(mtx-mn).^2));
  res = res  / max((prod(size(mtx)) - 1),1);
end
