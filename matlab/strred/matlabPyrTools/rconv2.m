% RES = RCONV2(MTX1, MTX2, CTR)
%
% Convolution of two matrices, with boundaries handled via reflection
% about the edge pixels.  Result will be of size of LARGER matrix.
% 
% The origin of the smaller matrix is assumed to be its center.
% For even dimensions, the origin is determined by the CTR (optional) 
% argument:
%      CTR   origin
%       0     DIM/2      (default)
%       1     (DIM/2)+1  

% Eero Simoncelli, 6/96.

function c = rconv2(a,b,ctr)

if (exist('ctr') ~= 1)
  ctr = 0;
end

if (( size(a,1) >= size(b,1) ) & ( size(a,2) >= size(b,2) ))
    large = a; small = b;
elseif  (( size(a,1) <= size(b,1) ) & ( size(a,2) <= size(b,2) ))
    large = b; small = a;
else
  error('one arg must be larger than the other in both dimensions!');
end

ly = size(large,1);
lx = size(large,2);
sy = size(small,1);
sx = size(small,2);

%% These values are one less than the index of the small mtx that falls on 
%% the border pixel of the large matrix when computing the first 
%% convolution response sample:
sy2 = floor((sy+ctr-1)/2);
sx2 = floor((sx+ctr-1)/2);

% pad with reflected copies
clarge = [ 
    large(sy-sy2:-1:2,sx-sx2:-1:2), large(sy-sy2:-1:2,:), ...
	large(sy-sy2:-1:2,lx-1:-1:lx-sx2); ...
    large(:,sx-sx2:-1:2),    large,   large(:,lx-1:-1:lx-sx2); ...
    large(ly-1:-1:ly-sy2,sx-sx2:-1:2), ...
      large(ly-1:-1:ly-sy2,:), ...
      large(ly-1:-1:ly-sy2,lx-1:-1:lx-sx2) ];

c = conv2(clarge,small,'valid');

