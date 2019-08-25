% HEIGHT = maxPyrHt(IMSIZE, FILTSIZE)
%
% Compute maximum pyramid height for given image and filter sizes.
% Specifically: the number of corrDn operations that can be sequentially
% performed when subsampling by a factor of 2.

% Eero Simoncelli, 6/96.

function height = maxPyrHt(imsz, filtsz)

imsz = imsz(:);
filtsz = filtsz(:);

if any(imsz == 1) % 1D image
  imsz = prod(imsz);
  filtsz = prod(filtsz);
elseif any(filtsz == 1)              % 2D image, 1D filter
  filtsz = [filtsz(1); filtsz(1)];
end

if any(imsz < filtsz)
  height = 0;
else
  height = 1 + maxPyrHt( floor(imsz/2), filtsz ); 
end
