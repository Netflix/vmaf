% [LEV,IND] = spyrLev(PYR,INDICES,LEVEL)
%
% Access a level from a steerable pyramid.
% Return as an SxB matrix, B = number of bands, S = total size of a band.
% Also returns an Bx2 matrix containing dimensions of the subbands.

% Eero Simoncelli, 6/96.

function [lev,ind] =  spyrLev(pyr,pind,level)

nbands = spyrNumBands(pind);
		
if ((level > spyrHt(pind)) | (level < 1))
  error(sprintf('Level number must be in the range [1, %d].', spyrHt(pind)));
end	
	
firstband = 2 + nbands*(level-1);
firstind = 1;
for l=1:firstband-1
  firstind = firstind + prod(pind(l,:));
end

ind = pind(firstband:firstband+nbands-1,:);
lev  = pyr(firstind:firstind+sum(prod(ind'))-1);
