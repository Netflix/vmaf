% [LEV,IND] = wpyrLev(PYR,INDICES,LEVEL)
%
% Access a level from a separable QMF/wavelet pyramid.
% Return as an SxB matrix, B = number of bands, S = total size of a band.
% Also returns an Bx2 matrix containing dimensions of the subbands.

% Eero Simoncelli, 6/96.

function [lev,ind] =  wpyrLev(pyr,pind,level)

if ((pind(1,1) == 1) | (pind(1,2) ==1))
  nbands = 1;
else
  nbands = 3;
end
		
if ((level > wpyrHt(pind)) | (level < 1))
  error(sprintf('Level number must be in the range [1, %d].', wpyrHt(pind)));
end	
	
firstband = 1 + nbands*(level-1)
firstind = 1;
for l=1:firstband-1
  firstind = firstind + prod(pind(l,:));
end


ind = pind(firstband:firstband+nbands-1,:);
lev  = pyr(firstind:firstind+sum(prod(ind'))-1);

