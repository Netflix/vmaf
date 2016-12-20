% RES = ZCONV2(MTX1, MTX2, CTR)
%
% Convolution of two matrices, with boundaries handled as if the larger mtx
% lies in a sea of zeros. Result will be of size of LARGER vector.
% 
% The origin of the smaller matrix is assumed to be its center.
% For even dimensions, the origin is determined by the CTR (optional) 
% argument:
%      CTR   origin
%       0     DIM/2      (default)
%       1     (DIM/2)+1  (behaves like conv2(mtx1,mtx2,'same'))

% Eero Simoncelli, 2/97.

function c = zconv2(a,b,ctr)

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

%% These values are the index of the small mtx that falls on the
%% border pixel of the large matrix when computing the first
%% convolution response sample:
sy2 = floor((sy+ctr+1)/2);
sx2 = floor((sx+ctr+1)/2);

clarge = conv2(large,small);
c = clarge(sy2:ly+sy2-1, sx2:lx+sx2-1);
