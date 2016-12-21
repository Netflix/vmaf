% [PYR, INDICES] = buildWpyr(IM, HEIGHT, FILT, EDGES)
%
% Construct a separable orthonormal QMF/wavelet pyramid on matrix (or vector) IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is maxPyrHt(IM,FILT).  You can also specify 'auto' to use this value.
%
% FILT (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Filter can be of even or odd length, but should be symmetric. 
% Default = 'qmf9'.  EDGES specifies edge-handling, and
% defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.

% Eero Simoncelli, 6/96.

function [pyr,pind] = buildWpyr(im, ht, filt, edges)

if (nargin < 1)
  error('First argument (IM) is required');
end

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('filt') ~= 1)
  filt = 'qmf9';
end

if (exist('edges') ~= 1)
  edges= 'reflect1';
end

if isstr(filt)
  filt = namedFilter(filt);
end

if ( (size(filt,1) > 1) & (size(filt,2) > 1) )
  error('FILT should be a 1D filter (i.e., a vector)');
else
  filt = filt(:);
end

hfilt = modulateFlip(filt);

% Stagger sampling if filter is odd-length:
if (mod(size(filt,1),2) == 0)
  stag = 2;
else
  stag = 1;
end

im_sz = size(im);

max_ht = maxPyrHt(im_sz, size(filt,1));
if ( (exist('ht') ~= 1) | (ht == 'auto') )
  ht = max_ht;
else
  if (ht > max_ht)
    error(sprintf('Cannot build pyramid higher than %d levels.',max_ht));
  end
end

if (ht <= 0)

  pyr = im(:);
  pind = im_sz;

else

  if (im_sz(2) == 1)
    lolo = corrDn(im, filt, edges, [2 1], [stag 1]);
    hihi = corrDn(im, hfilt, edges, [2 1], [2 1]);
  elseif (im_sz(1) == 1)
    lolo = corrDn(im, filt', edges, [1 2], [1 stag]);
    hihi = corrDn(im, hfilt', edges, [1 2], [1 2]);
  else
    lo = corrDn(im, filt, edges, [2 1], [stag 1]);
    hi = corrDn(im, hfilt, edges, [2 1], [2 1]);
    lolo = corrDn(lo, filt', edges, [1 2], [1 stag]);
    lohi = corrDn(hi, filt', edges, [1 2], [1 stag]); % horizontal
    hilo = corrDn(lo, hfilt', edges, [1 2], [1 2]); % vertical
    hihi = corrDn(hi, hfilt', edges, [1 2], [1 2]); % diagonal
  end

  [npyr,nind] = buildWpyr(lolo, ht-1, filt, edges);

  if ((im_sz(1) == 1) | (im_sz(2) == 1))
    pyr = [hihi(:); npyr];
    pind = [size(hihi); nind];
  else
    pyr = [lohi(:); hilo(:); hihi(:); npyr];
    pind = [size(lohi); size(hilo); size(hihi); nind];
  end

end
  
