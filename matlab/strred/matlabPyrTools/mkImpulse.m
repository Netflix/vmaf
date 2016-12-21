% IM = mkImpulse(SIZE, ORIGIN, AMPLITUDE)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing a single non-zero entry, at position ORIGIN (defaults to
% ceil(size/2)), of value AMPLITUDE (defaults to 1).

% Eero Simoncelli, 6/96.

function [res] = mkImpulse(sz, origin, amplitude)

sz = sz(:)';
if (size(sz,2) == 1)
  sz = [sz sz];
end

if (exist('origin') ~= 1)
  origin = ceil(sz/2);
end

if (exist('amplitude') ~= 1)
  amplitude = 1;
end

res = zeros(sz);
res(origin(1),origin(2)) = amplitude;
