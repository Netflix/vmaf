% NEWPYR = setPyrBand(PYR, INDICES, BAND, BAND_NUM)
%
% Insert an image (BAND) into a pyramid (gaussian, laplacian, QMF/wavelet, 
% or steerable).  Subbands are numbered consecutively, from finest
% (highest spatial frequency) to coarsest (lowest spatial frequency).

% Eero Simoncelli, 1/03.

function pyr =  pyrBand(pyr, pind, band, bandNum)

%% Check: PIND a valid index matrix?
if ( ~(ndims(pind) == 2) | ~(size(pind,2) == 2) | ~all(pind==round(pind)) )
  pind
  error('pyrTools:badArg',...
      'PIND argument is not an Nbands X 2 matrix of integers');
end

%% Check: PIND consistent with size of PYR?
if ( length(pyr) ~= sum(prod(pind,2)) )
  error('pyrTools:badPyr',...
      'Pyramid data vector length is inconsistent with index matrix PIND');
end

%% Check: size of BAND  consistent with desired BANDNUM?
if (~all(size(band) == pind(bandNum,:)))
  size(band)
  pind(bandNum,:)
  error('pyrTools:badArg',...
      'size of BAND to be inserted is inconsistent with BAND_NUM');
end

pyr(pyrBandIndices(pind,bandNum)) = vectify(band); 
