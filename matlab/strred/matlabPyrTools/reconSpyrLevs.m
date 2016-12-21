% RES = reconSpyrLevs(PYR,INDICES,LOFILT,BFILTS,EDGES,LEVS,BANDS)
%
% Recursive function for reconstructing levels of a steerable pyramid
% representation.  This is called by reconSpyr, and is not usually
% called directly.

% Eero Simoncelli, 6/96.

function res = reconSpyrLevs(pyr,pind,lofilt,bfilts,edges,levs,bands);

nbands = size(bfilts,2);
lo_ind = nbands+1;
res_sz = pind(1,:);

% Assume square filters:
bfiltsz =  round(sqrt(size(bfilts,1)));

if any(levs > 1)

  if  (size(pind,1) > lo_ind)
    nres = reconSpyrLevs( pyr(1+sum(prod(pind(1:lo_ind-1,:)')):size(pyr,1)),  ...
	pind(lo_ind:size(pind,1),:), ...
	lofilt, bfilts, edges, levs-1, bands);
  else
    nres = pyrBand(pyr,pind,lo_ind); 	% lowpass subband
  end

  res = upConv(nres, lofilt, edges, [2 2], [1 1], res_sz);

else

  res = zeros(res_sz);

end
	
if any(levs == 1)
  ind = 1;
  for b = 1:nbands
    if any(bands == b)
      bfilt = reshape(bfilts(:,b), bfiltsz, bfiltsz);
      upConv(reshape(pyr(ind:ind+prod(res_sz)-1), res_sz(1), res_sz(2)), ...
		  bfilt, edges, [1 1], [1 1], res_sz, res);
    end
    ind = ind + prod(res_sz);
  end
end
