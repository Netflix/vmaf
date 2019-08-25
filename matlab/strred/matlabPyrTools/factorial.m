%% RES = factorial(NUM)
%
% Factorial function that works on matrices (matlab's does not).

% EPS, 11/02

function res = factorial(num)

res = ones(size(num));

ind = find(num > 0);
if ( ~isempty(ind) )
  subNum = num(ind);
  res(ind) = subNum .* factorial(subNum-1);
end

