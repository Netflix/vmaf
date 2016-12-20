% RES = reconWpyr(PYR, INDICES, FILT, EDGES, LEVS, BANDS)
%
% Reconstruct image from its separable orthonormal QMF/wavelet pyramid
% representation, as created by buildWpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% FILT (optional) can be a string naming a standard filter (see
% namedFilter), or a vector which will be used for (separable)
% convolution.  Default = 'qmf9'.  EDGES specifies edge-handling,
% and defaults to 'reflect1' (see corrDn).
%
% LEVS (optional) should be a vector of levels to include, or the string
% 'all' (default).  1 corresponds to the finest scale.  The lowpass band
% corresponds to wpyrHt(INDICES)+1.
%
% BANDS (optional) should be a vector of bands to include, or the string
% 'all' (default).   1=horizontal, 2=vertical, 3=diagonal.  This is only used
% for pyramids of 2D images.

% Eero Simoncelli, 6/96.

function res = reconWpyr(pyr, ind, filt, edges, levs, bands)

if (nargin < 2)
  error('First two arguments (PYR INDICES) are required');
end

%%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('filt') ~= 1)
  filt = 'qmf9';
end

if (exist('edges') ~= 1)
  edges= 'reflect1';
end

if (exist('levs') ~= 1)
  levs = 'all';
end

if (exist('bands') ~= 1)
  bands = 'all';
end

%%------------------------------------------------------------

maxLev = 1+wpyrHt(ind);
if strcmp(levs,'all')
  levs = [1:maxLev]';
else
  if (any(levs > maxLev))
    error(sprintf('Level numbers must be in the range [1, %d].', maxLev));
  end
  levs = levs(:);
end

if strcmp(bands,'all')
  bands = [1:3]';
else
  if (any(bands < 1) | any(bands > 3))
    error('Band numbers must be in the range [1,3].');
  end
  bands = bands(:);
end

if isstr(filt)
  filt = namedFilter(filt);
end

filt = filt(:);
hfilt = modulateFlip(filt);

%% For odd-length filters, stagger the sampling lattices:
if (mod(size(filt,1),2) == 0)
	stag = 2;
else
	stag = 1;
end

%% Compute size of result image: assumes critical sampling (boundaries correct)
res_sz = ind(1,:);
if (res_sz(1) == 1)
  loind = 2;
  res_sz(2) = sum(ind(:,2));
elseif (res_sz(2) == 1)	
  loind = 2;
  res_sz(1) = sum(ind(:,1));
else
  loind = 4;
  res_sz = ind(1,:) + ind(2,:);  %%horizontal + vertical bands.
  hres_sz = [ind(1,1), res_sz(2)];
  lres_sz = [ind(2,1), res_sz(2)];
end
	

%% First, recursively collapse coarser scales:
if any(levs > 1)  

  if (size(ind,1) > loind)
    nres = reconWpyr( pyr(1+sum(prod(ind(1:loind-1,:)')):size(pyr,1)), ...
	ind(loind:size(ind,1),:), filt, edges, levs-1, bands);
  else
    nres = pyrBand(pyr, ind, loind); 	% lowpass subband
  end

  if (res_sz(1) == 1)
    res = upConv(nres, filt', edges, [1 2], [1 stag], res_sz);
  elseif (res_sz(2) == 1)
    res = upConv(nres, filt, edges, [2 1], [stag 1], res_sz);
  else
    ires = upConv(nres, filt', edges, [1 2], [1 stag], lres_sz); 
    res = upConv(ires, filt, edges, [2 1], [stag 1], res_sz);
  end
  
else

  res = zeros(res_sz);

end

	
%% Add  in reconstructed bands from this level:
if any(levs == 1)
  if (res_sz(1) == 1)
    upConv(pyrBand(pyr,ind,1), hfilt', edges, [1 2], [1 2], res_sz, res);
  elseif (res_sz(2) == 1)
    upConv(pyrBand(pyr,ind,1), hfilt, edges, [2 1], [2 1], res_sz, res);
  else
    if any(bands == 1) % horizontal
      ires = upConv(pyrBand(pyr,ind,1),filt',edges,[1 2],[1 stag],hres_sz);
      upConv(ires,hfilt,edges,[2 1],[2 1],res_sz,res);  %destructively modify res
    end
    if any(bands == 2) % vertical
      ires = upConv(pyrBand(pyr,ind,2),hfilt',edges,[1 2],[1 2],lres_sz);
      upConv(ires,filt,edges,[2 1],[stag 1],res_sz,res);  %destructively modify res
    end
    if any(bands == 3) % diagonal
      ires =  upConv(pyrBand(pyr,ind,3),hfilt',edges,[1 2],[1 2],hres_sz);
      upConv(ires,hfilt,edges,[2 1],[2 1],res_sz,res);  %destructively modify res
    end
  end
end
  
