% RES = pyrBandIndices(INDICES, BAND_NUM)
%
% Return indices for accessing a subband from a pyramid 
% (gaussian, laplacian, QMF/wavelet, steerable).

% Eero Simoncelli, 6/96.

function indices =  pyrBandIndices(pind,band)

if ((band > size(pind,1)) | (band < 1))
  error(sprintf('BAND_NUM must be between 1 and number of pyramid bands (%d).', ...
      size(pind,1)));
end

if (size(pind,2) ~= 2)
  error('INDICES must be an Nx2 matrix indicating the size of the pyramid subbands');
end

ind = 1;
for l=1:band-1
  ind = ind + prod(pind(l,:));
end

indices = ind:ind+prod(pind(band,:))-1;
