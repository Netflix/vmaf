% [HEIGHT] = wpyrHt(INDICES)
%
% Compute height of separable QMF/wavelet pyramid with given index matrix.

% Eero Simoncelli, 6/96.

function [ht] =  wpyrHt(pind)

if ((pind(1,1) == 1) | (pind(1,2) ==1))
	nbands = 1;
else
	nbands = 3;
end

ht = (size(pind,1)-1)/nbands;
