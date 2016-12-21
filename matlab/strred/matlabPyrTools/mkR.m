% IM = mkR(SIZE, EXPT, ORIGIN)
% 
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a radial ramp function, raised to power EXPT
% (default = 1), with given ORIGIN (default = (size+1)/2, [1 1] =
% upper left).  All but the first argument are optional.

% Eero Simoncelli, 6/96.

function [res] = mkR(sz, expt, origin)

sz = sz(:);
if (size(sz,1) == 1)
  sz = [sz,sz];
end
 
% -----------------------------------------------------------------
% OPTIONAL args:

if (exist('expt') ~= 1)
  expt = 1;
end

if (exist('origin') ~= 1)
  origin = (sz+1)/2;
end

% -----------------------------------------------------------------

[xramp,yramp] = meshgrid( [1:sz(2)]-origin(2), [1:sz(1)]-origin(1) );

res = (xramp.^2 + yramp.^2).^(expt/2);
