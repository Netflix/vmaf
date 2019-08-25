% K = KURT2(MTX,MEAN,VAR)
%
% Sample kurtosis (fourth moment divided by squared variance) 
% of a matrix.  Kurtosis of a Gaussian distribution is 3.
%  MEAN (optional) and VAR (optional) make the computation faster.

% Eero Simoncelli, 6/96.

function res = kurt2(mtx, mn, v)

if (exist('mn') ~= 1)
	mn =  mean(mean(mtx));
end

if (exist('v') ~= 1)
	v =  var2(mtx,mn);
end

if (isreal(mtx))
  res = mean(mean(abs(mtx-mn).^4)) / (v^2);
else
  res = mean(mean(real(mtx-mn).^4)) / (real(v)^2) + ...
      i*mean(mean(imag(mtx-mn).^4)) / (imag(v)^2);
end
