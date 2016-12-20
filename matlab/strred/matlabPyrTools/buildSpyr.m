% [PYR, INDICES, STEERMTX, HARMONICS] = buildSpyr(IM, HEIGHT, FILTFILE, EDGES)
%
% Construct a steerable pyramid on matrix IM.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is maxPyrHt(size(IM),size(FILT)). 
% You can also specify 'auto' to use this value.
%
% FILTFILE (optional) should be a string referring to an m-file that
% returns the rfilters.  (examples: 'sp0Filters', 'sp1Filters',
% 'sp3Filters','sp5Filters'.  default = 'sp1Filters'). EDGES specifies
% edge-handling, and defaults to 'reflect1' (see corrDn).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
% See the function STEER for a description of STEERMTX and HARMONICS.

% Eero Simoncelli, 6/96.
% See http://www.cis.upenn.edu/~eero/steerpyr.html for more
% information about the Steerable Pyramid image decomposition.

function [pyr,pind,steermtx,harmonics] = buildSpyr(im, ht, filtfile, edges)

%-----------------------------------------------------------------
%% DEFAULTS:

if (exist('filtfile') ~= 1)
  filtfile = 'sp1Filters';
end

if (exist('edges') ~= 1)
  edges= 'reflect1';
end

if (isstr(filtfile) & (exist(filtfile) == 2))
   [lo0filt,hi0filt,lofilt,bfilts,steermtx,harmonics] = eval(filtfile);
else
  fprintf(1,'\nUse buildSFpyr for pyramids with arbitrary numbers of orientation bands.\n');
  error('FILTFILE argument must be the name of an M-file containing SPYR filters.');
end

max_ht = maxPyrHt(size(im), size(lofilt,1));
if ( (exist('ht') ~= 1) | (ht == 'auto') )
  ht = max_ht;
else
  if (ht > max_ht)
    error(sprintf('Cannot build pyramid higher than %d levels.',max_ht));
  end
end

%-----------------------------------------------------------------

hi0 = corrDn(im, hi0filt, edges);
lo0 = corrDn(im, lo0filt, edges);

[pyr,pind] = buildSpyrLevs(lo0, ht, lofilt, bfilts, edges);

pyr = [hi0(:) ; pyr];
pind = [size(hi0); pind];
  
