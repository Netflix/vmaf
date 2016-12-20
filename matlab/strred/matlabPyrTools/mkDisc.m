% IM = mkDisc(SIZE, RADIUS, ORIGIN, TWIDTH, VALS)
%
% Make a "disk" image.  SIZE specifies the matrix size, as for
% zeros().  RADIUS (default = min(size)/4) specifies the radius of 
% the disk.  ORIGIN (default = (size+1)/2) specifies the 
% location of the disk center.  TWIDTH (in pixels, default = 2) 
% specifies the width over which a soft threshold transition is made.
% VALS (default = [0,1]) should be a 2-vector containing the
% intensity value inside and outside the disk.  

% Eero Simoncelli, 6/96.

function [res] = mkDisc(sz, rad, origin, twidth, vals)

if (nargin < 1)
  error('Must pass at least a size argument');
end
  
sz = sz(:);
if (size(sz,1) == 1)
  sz = [sz sz];
end
 
%------------------------------------------------------------
% OPTIONAL ARGS:

if (exist('rad') ~= 1)
  rad = min(sz(1),sz(2))/4;
end

if (exist('origin') ~= 1)
  origin = (sz+1)./2;
end

if (exist('twidth') ~= 1)
  twidth = 2;
end

if (exist('vals') ~= 1)
  vals = [1,0];
end

%------------------------------------------------------------

res = mkR(sz,1,origin);

if (abs(twidth) < realmin)
  res = vals(2) + (vals(1) - vals(2)) * (res <= rad);
else
  [Xtbl,Ytbl] = rcosFn(twidth, rad, [vals(1), vals(2)]);
  res = pointOp(res, Ytbl, Xtbl(1), Xtbl(2)-Xtbl(1), 0);
% 
% OLD interp1 VERSION:
%  res = res(:);
%  Xtbl(1) = min(res);
%  Xtbl(size(Xtbl,2)) = max(res);
%  res = reshape(interp1(Xtbl,Ytbl,res), sz(1), sz(2));
% 
end
  

