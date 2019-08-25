% IM = mkRamp(SIZE, DIRECTION, SLOPE, INTERCEPT, ORIGIN)
%
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a ramp function, with given gradient DIRECTION
% (radians, CW from X-axis, default = 0), SLOPE (per pixel, default =
% 1), and a value of INTERCEPT (default = 0) at the ORIGIN (default =
% (size+1)/2, [1 1] = upper left).  All but the first argument are
% optional.

% Eero Simoncelli, 6/96. 2/97: adjusted coordinate system.

function [res] = mkRamp(sz, dir, slope, intercept, origin)

sz = sz(:);
if (size(sz,1) == 1)
  sz = [sz,sz];
end

% -----------------------------------------------------------------
% OPTIONAL args:

if (exist('dir') ~= 1)
  dir = 0;
end
 
if (exist('slope') ~= 1)
  slope = 1;
end
 
if (exist('intercept') ~= 1)
  intercept = 0;
end

if (exist('origin') ~= 1)
  origin = (sz+1)/2;
end

% -----------------------------------------------------------------

xinc = slope*cos(dir);
yinc = slope*sin(dir);

[xramp,yramp] = meshgrid( xinc*([1:sz(2)]-origin(2)), ...
    yinc*([1:sz(1)]-origin(1)) );
 
res = intercept + xramp + yramp;

