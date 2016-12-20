% [HEIGHT] = spyrHt(INDICES)
%
% Compute height of steerable pyramid with given index matrix.

% Eero Simoncelli, 6/96.

function [ht] =  spyrHt(pind)

nbands = spyrNumBands(pind);

% Don't count lowpass, or highpass residual bands
if (size(pind,1) > 2)
  ht = (size(pind,1)-2)/nbands;
else
  ht = 0;
end
