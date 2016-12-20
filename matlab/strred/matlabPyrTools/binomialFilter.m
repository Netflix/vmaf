% KERNEL = binomialFilter(size)
%
% Returns a vector of binomial coefficients of order (size-1) .

% Eero Simoncelli, 2/97.

function [kernel] = binomialFilter(sz)

if (sz < 2)
  error('size argument must be larger than 1');
end

kernel = [0.5 0.5]';

for n=1:sz-2
  kernel = conv([0.5 0.5]', kernel);
end
  
