% [PYR, INDICES] = buildSpyrLevs(LOIM, HEIGHT, LOFILT, BFILTS, EDGES)
%
% Recursive function for constructing levels of a steerable pyramid.  This
% is called by buildSpyr, and is not usually called directly.

% Eero Simoncelli, 6/96.

function [pyr,pind] = buildSpyrLevs(lo0,ht,lofilt,bfilts,edges);

if (ht <= 0)

  pyr = lo0(:);
  pind = size(lo0);

else

  % Assume square filters:
  bfiltsz =  round(sqrt(size(bfilts,1)));

  bands = zeros(prod(size(lo0)),size(bfilts,2));
  bind = zeros(size(bfilts,2),2);

  for b = 1:size(bfilts,2)
    filt = reshape(bfilts(:,b),bfiltsz,bfiltsz);
    band = corrDn(lo0, filt, edges);
    bands(:,b) = band(:);
    bind(b,:)  = size(band);
  end
	
  lo = corrDn(lo0, lofilt, edges, [2 2], [1 1]);
  
  [npyr,nind] = buildSpyrLevs(lo, ht-1, lofilt, bfilts, edges);

  pyr = [bands(:); npyr];
  pind = [bind; nind];
	
end
