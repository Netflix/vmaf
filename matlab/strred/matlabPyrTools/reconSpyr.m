% RES = reconSpyr(PYR, INDICES, FILTFILE, EDGES, LEVS, BANDS)
%
% Reconstruct image from its steerable pyramid representation, as created
% by buildSpyr.
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
%
% FILTFILE (optional) should be a string referring to an m-file that returns
% the rfilters.  examples: sp0Filters, sp1Filters, sp3Filters 
% (default = 'sp1Filters'). 
% EDGES specifies edge-handling, and defaults to 'reflect1' (see
% corrDn).
% 
% LEVS (optional) should be a list of levels to include, or the string
% 'all' (default).  0 corresonds to the residual highpass subband.  
% 1 corresponds to the finest oriented scale.  The lowpass band
% corresponds to number spyrHt(INDICES)+1.
%
% BANDS (optional) should be a list of bands to include, or the string
% 'all' (default).  1 = vertical, rest proceeding anti-clockwise.

% Eero Simoncelli, 6/96.

function res = reconSpyr(pyr, pind, filtfile, edges, levs, bands)

%%------------------------------------------------------------
%% DEFAULTS:

if (exist('filtfile') ~= 1)
  filtfile = 'sp1Filters';
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

if (isstr(filtfile) & (exist(filtfile) == 2))
   [lo0filt,hi0filt,lofilt,bfilts,steermtx,harmonics] = eval(filtfile);
   nbands = spyrNumBands(pind);
   if ((nbands > 0) & (size(bfilts,2) ~= nbands))
     error('Number of pyramid bands is inconsistent with filter file');
   end
else
  error('filtfile argument must be the name of an M-file containing SPYR filters.');
end

maxLev =  1+spyrHt(pind);
if strcmp(levs,'all')
  levs = [0:maxLev]';
else
  if (any(levs > maxLev) | any(levs < 0))
    error(sprintf('Level numbers must be in the range [0, %d].', maxLev));
  end
  levs = levs(:);
end

if strcmp(bands,'all')
  bands = [1:nbands]';
else
  if (any(bands < 1) | any(bands > nbands))
    error(sprintf('Band numbers must be in the range [1,3].', nbands));
  end
  bands = bands(:);
end

if (spyrHt(pind) == 0)
  if (any(levs==1))
    res1 = pyrBand(pyr,pind,2);
  else
    res1 = zeros(pind(2,:));
  end
else
  res1 = reconSpyrLevs(pyr(1+prod(pind(1,:)):size(pyr,1)), ...
      pind(2:size(pind,1),:), ...
      lofilt, bfilts, edges, levs, bands);
end

res = upConv(res1, lo0filt, edges);

%% residual highpass subband
if any(levs == 0)
   upConv( subMtx(pyr, pind(1,:)), hi0filt, edges, [1 1], [1 1], size(res), res);
end
 
