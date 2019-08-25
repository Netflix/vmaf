% RES = wpyrBand(PYR, INDICES, LEVEL, BAND)
%
% Access a subband from a separable QMF/wavelet pyramid.  
% 
% LEVEL (optional, default=1) indicates the scale (finest = 1,
% coarsest = wpyrHt(INDICES)).  
% 
% BAND (optional, default=1) indicates which subband (1=horizontal,
% 2=vertical, 3=diagonal).

% Eero Simoncelli, 6/96.

function im =  wpyrBand(pyr,pind,level,band)

if (exist('level') ~= 1)
  level = 1;
end

if (exist('band') ~= 1)
  band = 1;
end

if ((pind(1,1) == 1) | (pind(1,2) ==1))
  nbands = 1;
else
  nbands = 3;
end
		
if ((band > nbands) | (band < 1))
  error(sprintf('Bad band number (%d) should be in range [1,%d].', band, nbands));
end
	
maxLev = wpyrHt(pind);
if ((level > maxLev) | (level < 1))
  error(sprintf('Bad level number (%d), should be in range [1,%d].', level, maxLev));
end

band = band + nbands*(level-1);
im = pyrBand(pyr,pind,band);
