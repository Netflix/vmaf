% S = SKEW2(MTX,MEAN,VAR)
%
% Sample skew (third moment divided by variance^3/2) of a matrix.
%  MEAN (optional) and VAR (optional) make the computation faster.

function res = skew2(mtx, mn, v)

if (exist('mn') ~= 1)
  mn =  mean2(mtx);
end

if (exist('v') ~= 1)
  v =  var2(mtx,mn);
end

if (isreal(mtx))
  res = mean(mean((mtx-mn).^3)) / (v^(3/2));
else
  res = mean(mean(real(mtx-mn).^3)) / (real(v)^(3/2)) + ...
      i * mean(mean(imag(mtx-mn).^3)) / (imag(v)^(3/2));
end
