% [PYR, INDICES] = buildGpyr(IM, HEIGHT, FILT, EDGES)
%
% Construct a Gaussian pyramid on matrix IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is 1+maxPyrHt(size(IM),size(FILT)). 
% You can also specify 'auto' to use this value.
%
% FILT (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'binom5'.  EDGES specifies edge-handling, and
% defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.

% Eero Simoncelli, 6/96.

function [pyr,pind] = buildGpyr(im, ht, filt, edges)

if (nargin < 1)
  error('First argument (IM) is required');
end

im_sz = size(im);

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('filt') ~= 1)
  filt = 'binom5';
end

if isstr(filt)
  filt = namedFilter(filt);
end

if ( (size(filt,1) > 1) & (size(filt,2) > 1) )
  error('FILT should be a 1D filter (i.e., a vector)');
else
  filt = filt(:);
end

max_ht = 1 + maxPyrHt(im_sz, size(filt,1));
if ( (exist('ht') ~= 1) | (ht == 'auto') )
  ht = max_ht;
else
  if (ht > max_ht)
    error(sprintf('Cannot build pyramid higher than %d levels.',max_ht));
  end
end

if (exist('edges') ~= 1)
  edges= 'reflect1';
end

%------------------------------------------------------------

if (ht <= 1)

  pyr = im(:);
  pind = im_sz;

else

  if (im_sz(2) == 1)
    lo2 = corrDn(im, filt, edges, [2 1], [1 1]);
  elseif (im_sz(1) == 1)
    lo2 = corrDn(im, filt', edges, [1 2], [1 1]);
  else
    lo = corrDn(im, filt', edges, [1 2], [1 1]);
    lo2 = corrDn(lo, filt, edges, [2 1], [1 1]);
  end
  
  [npyr,nind] = buildGpyr(lo2, ht-1, filt, edges);

  pyr = [im(:); npyr];
  pind = [im_sz; nind];
  
end
  
