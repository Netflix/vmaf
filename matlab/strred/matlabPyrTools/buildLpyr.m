% [PYR, INDICES] = buildLpyr(IM, HEIGHT, FILT1, FILT2, EDGES)
%
% Construct a Laplacian pyramid on matrix (or vector) IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is 1+maxPyrHt(size(IM),size(FILT)).  You can also specify 'auto' to
% use this value.
%
% FILT1 (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'binom5'.  FILT2 specifies the "expansion"
% filter (default = filt1).  EDGES specifies edge-handling, and
% defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.

% Eero Simoncelli, 6/96.

function [pyr,pind] = buildLpyr(im, ht, filt1, filt2, edges)

if (nargin < 1)
  error('First argument (IM) is required');
end

im_sz = size(im);

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('filt1') ~= 1)
  filt1 = 'binom5';
end
 
if isstr(filt1)
  filt1 = namedFilter(filt1);
end

if ( (size(filt1,1) > 1) & (size(filt1,2) > 1) )
  error('FILT1 should be a 1D filter (i.e., a vector)');
else
  filt1 = filt1(:);
end

if (exist('filt2') ~= 1)
  filt2 = filt1;
end

if isstr(filt2)
  filt2 = namedFilter(filt2);
end

if ( (size(filt2,1) > 1) & (size(filt2,2) > 1) )
  error('FILT2 should be a 1D filter (i.e., a vector)');
else
  filt2 = filt2(:);
end

max_ht = 1 + maxPyrHt(im_sz, max(size(filt1,1), size(filt2,1)));
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
    lo2 = corrDn(im, filt1, edges, [2 1], [1 1]);
  elseif (im_sz(1) == 1)
    lo2 = corrDn(im, filt1', edges, [1 2], [1 1]);
  else
    lo = corrDn(im, filt1', edges, [1 2], [1 1]);
    int_sz = size(lo);
    lo2 = corrDn(lo, filt1, edges, [2 1], [1 1]);
  end

  [npyr,nind] = buildLpyr(lo2, ht-1, filt1, filt2, edges);

  if (im_sz(1) == 1)
    hi2 = upConv(lo2, filt2', edges, [1 2], [1 1], im_sz);
  elseif (im_sz(2) == 1)
    hi2 = upConv(lo2, filt2, edges, [2 1], [1 1], im_sz);
  else
    hi = upConv(lo2, filt2, edges, [2 1], [1 1], int_sz);
    hi2 = upConv(hi, filt2', edges, [1 2], [1 1], im_sz);
  end

  hi2 = im - hi2;

  pyr = [hi2(:); npyr];
  pind = [im_sz; nind];

end
  
