% IM = mkAngle(SIZE, PHASE, ORIGIN)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of the polar angle (in radians, CW from the
% X-axis, ranging from -pi to pi), relative to angle PHASE (default =
% 0), about ORIGIN pixel (default = (size+1)/2).

% Eero Simoncelli, 6/96.

function [res] = mkAngle(sz, phase, origin)

sz = sz(:);
if (size(sz,1) == 1)
  sz = [sz,sz];
end

% -----------------------------------------------------------------
% OPTIONAL args:

if (exist('origin') ~= 1)
  origin = (sz+1)/2;
end

% -----------------------------------------------------------------

[xramp,yramp] = meshgrid( [1:sz(2)]-origin(2), [1:sz(1)]-origin(1) );

res = atan2(yramp,xramp);

if (exist('phase') == 1)
  res = mod(res+(pi-phase),2*pi)-pi;
end
