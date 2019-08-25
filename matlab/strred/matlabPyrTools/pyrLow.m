% RES = pyrLow(PYR, INDICES)
%
% Access the lowpass subband from a pyramid 
%   (gaussian, laplacian, QMF/wavelet, steerable).

% Eero Simoncelli, 6/96.

function res =  pyrLow(pyr,pind)

band = size(pind,1);

res = reshape( pyr(pyrBandIndices(pind,band)), pind(band,1), pind(band,2) );
