% [NBANDS] = spyrNumBands(INDICES)
%
% Compute number of orientation bands in a steerable pyramid with
% given index matrix.  If the pyramid contains only the highpass and
% lowpass bands (i.e., zero levels), returns 0.

% Eero Simoncelli, 2/97.

function [nbands] =  spyrNumBands(pind)

if (size(pind,1) == 2)
  nbands  = 0;
else
  % Count number of orientation bands:
  b = 3;
  while ((b <= size(pind,1)) & all( pind(b,:) == pind(2,:)) )
    b = b+1;
  end
  nbands = b-2;
end
