% RES = reconLpyr(PYR, INDICES, LEVS, FILT2, EDGES)
%
% Reconstruct image from Laplacian pyramid, as created by buildLpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% LEVS (optional) should be a list of levels to include, or the string
% 'all' (default).  The finest scale is number 1.  The lowpass band
% corresponds to lpyrHt(INDICES)+1.
%
% FILT2 (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'binom5'.  EDGES specifies edge-handling,
% and defaults to 'reflect1' (see corrDn).

% Eero Simoncelli, 6/96

function res = reconLpyr(pyr, ind, levs, filt2, edges)

if (nargin < 2)
  error('First two arguments (PYR, INDICES) are required');
end
  
%%------------------------------------------------------------
%% DEFAULTS:

if (exist('levs') ~= 1)
  levs = 'all';
end

if (exist('filt2') ~= 1)
  filt2 = 'binom5';
end

if (exist('edges') ~= 1)
  edges= 'reflect1';
end
%%------------------------------------------------------------

maxLev =  1+lpyrHt(ind);
if strcmp(levs,'all')
  levs = [1:maxLev]';
else
  if (any(levs > maxLev))
    error(sprintf('Level numbers must be in the range [1, %d].', maxLev));
  end
  levs = levs(:);
end

if isstr(filt2)
  filt2 = namedFilter(filt2);
end

filt2 = filt2(:);
res_sz = ind(1,:);

if any(levs > 1)

  int_sz = [ind(1,1), ind(2,2)];
  
  nres = reconLpyr( pyr(prod(res_sz)+1:size(pyr,1)), ...
      ind(2:size(ind,1),:), levs-1, filt2, edges);
  
  if (res_sz(1) == 1)
    res = upConv(nres, filt2', edges, [1 2], [1 1], res_sz);
  elseif (res_sz(2) == 1)
    res = upConv(nres, filt2, edges, [2 1], [1 1], res_sz);
  else
    hi = upConv(nres, filt2, edges, [2 1], [1 1], int_sz);
    res = upConv(hi, filt2', edges, [1 2], [1 1], res_sz);
  end

else
  
  res = zeros(res_sz);

end

if any(levs == 1)
  res = res + pyrBand(pyr,ind,1);
end
