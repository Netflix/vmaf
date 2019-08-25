% RANGE = showWpyr (PYR, INDICES, RANGE, GAP, LEVEL_SCALE_FACTOR)
% 
% Display a separable QMF/wavelet pyramid, specified by PYR and INDICES
% (see buildWpyr), in the current figure.
% 
% RANGE is a 2-vector specifying the values that map to black and
% white, respectively.  These values are scaled by
% LEVEL_SCALE_FACTOR^(lev-1) for bands at each level.  Passing a value
% of 'auto1' sets RANGE to the min and max values of MATRIX.  'auto2'
% sets RANGE to 3 standard deviations below and above 0.0.  In both of
% these cases, the lowpass band is independently scaled.  A value of
% 'indep1' sets the range of each subband independently, as in a call
% to showIm(subband,'auto1').  Similarly, 'indep2' causes each subband
% to be scaled independently as if by showIm(subband,'indep2').
% The default value for RANGE is 'auto1' for 1D images, and 'auto2' for
% 2D images.
% 
% GAP (optional, default=1) specifies the gap in pixels to leave
% between subbands (2D images only).  
% 
% LEVEL_SCALE_FACTOR indicates the relative scaling between pyramid
% levels.  This should be set to the sum of the kernel taps of the
% lowpass filter used to construct the pyramid (default assumes
% L2-normalized filters, using a value of 2 for 2D images, sqrt(2) for
% 1D images).

% Eero Simoncelli, 2/97.

function [range] = showWpyr(pyr, pind, range, gap, scale);

% Determine 1D or 2D pyramid:
if ((pind(1,1) == 1) | (pind(1,2) ==1))
  nbands = 1;
else
  nbands = 3;
end

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('range') ~= 1)  
  if (nbands==1)
    range = 'auto1';
  else
    range = 'auto2';
  end
end
		
if (exist('gap') ~= 1)
  gap = 1;
end

if (exist('scale') ~= 1)
  if (nbands == 1)
    scale = sqrt(2);
  else
    scale = 2;
  end
end

%------------------------------------------------------------

ht = wpyrHt(pind);
nind = size(pind,1);

%% Auto range calculations:
if strcmp(range,'auto1')
  range = zeros(nind,1);
  mn = 0.0; mx = 0.0;
  for lnum = 1:ht
    for bnum = 1:nbands
      band = wpyrBand(pyr,pind,lnum,bnum)/(scale^(lnum-1));
      range((lnum-1)*nbands+bnum) = scale^(lnum-1);
      [bmn,bmx] = range2(band);
      mn = min(mn, bmn);  mx = max(mx, bmx);
    end    
  end
  if (nbands == 1)
    pad = (mx-mn)/12;			% *** MAGIC NUMBER!!
    mn = mn-pad;  mx = mx+pad;
  end
  range = range * [mn mx]; 		% outer product
  band = pyrLow(pyr,pind);
  [mn,mx] = range2(band);
  if (nbands == 1)
    pad = (mx-mn)/12; 			% *** MAGIC NUMBER!!
    mn = mn-pad;  mx = mx+pad;
  end
  range(nind,:) = [mn, mx];

elseif strcmp(range,'indep1')
  range = zeros(nind,2);
  for bnum = 1:nind
    band = pyrBand(pyr,pind,bnum);
    [mn,mx] = range2(band);
    if (nbands == 1)
      pad = (mx-mn)/12; 		% *** MAGIC NUMBER!!
      mn = mn-pad;  mx = mx+pad;
    end
    range(bnum,:) =  [mn mx];
  end

elseif strcmp(range,'auto2')
  range = zeros(nind,1);
  sqsum = 0;  numpixels = 0;
  for lnum = 1:ht
    for bnum = 1:nbands
      band = wpyrBand(pyr,pind,lnum,bnum)/(scale^(lnum-1));
      sqsum = sqsum + sum(sum(band.^2));
      numpixels = numpixels + prod(size(band));
      range((lnum-1)*nbands+bnum) = scale^(lnum-1);
    end    
  end
  stdev = sqrt(sqsum/(numpixels-1));
  range = range * [ -3*stdev 3*stdev ]; % outer product
  band = pyrLow(pyr,pind);
  av = mean2(band);   stdev = sqrt(var2(band));
  range(nind,:) = [av-2*stdev,av+2*stdev];

elseif strcmp(range,'indep2')
  range = zeros(nind,2);
  for bnum = 1:(nind-1)
    band = pyrBand(pyr,pind,bnum);
    stdev = sqrt(var2(band));
    range(bnum,:) =  [ -3*stdev 3*stdev ];
  end
  band = pyrLow(pyr,pind);
  av = mean2(band);   stdev = sqrt(var2(band));
  range(nind,:) = [av-2*stdev,av+2*stdev];
  
elseif isstr(range)
  error(sprintf('Bad RANGE argument: %s',range))
  
elseif ((size(range,1) == 1) & (size(range,2) == 2))
  scales = scale.^[0:ht];
  if (nbands ~= 1)
    scales = [scales; scales; scales];
  end
  range = scales(:) * range;		% outer product
  band = pyrLow(pyr,pind);
  range(nind,:) = range(nind,:) + mean2(band) - mean(range(nind,:));

end

% CLEAR FIGURE:
clf;

if (nbands == 1)

  %%%%%  1D signal:
  for bnum=1:nind
    band = pyrBand(pyr,pind,bnum);
    subplot(nind,1,nind-bnum+1);
    plot(band);
    axis([1, prod(size(band)), range(bnum,:)]);
  end

else  

  %%%%% 2D signal:
  colormap(gray);
  cmap = get(gcf,'Colormap');
  nshades = size(cmap,1);

  %  Find background color index:
  clr = get(gcf,'Color');
  bg = 1;
  dist = norm(cmap(bg,:)-clr);
  for n = 1:nshades
    ndist = norm(cmap(n,:)-clr);
    if (ndist < dist)
      dist = ndist;
      bg = n;
    end
  end  

  %% Compute positions of subbands:
  llpos = ones(nind,2);
  for lnum = 1:ht
    ind1 = nbands*(lnum-1) + 1;
    xpos = pind(ind1,2) + 1 + gap*(ht-lnum+1);
    ypos = pind(ind1+1,1) + 1 + gap*(ht-lnum+1);
    llpos(ind1:ind1+2,:) = [ypos 1; 1 xpos; ypos xpos];
  end
  llpos(nind,:) = [1 1];  %lowpass

  %% Make position list positive, and allocate appropriate image:
  llpos = llpos - ones(nind,1)*min(llpos) + 1;
  urpos = llpos + pind - 1;
  d_im = bg + zeros(max(urpos));

  %% Paste bands into image, (im-r1)*(nshades-1)/(r2-r1) + 1.5 
  for bnum=1:nind
    mult = (nshades-1) / (range(bnum,2)-range(bnum,1));
    d_im(llpos(bnum,1):urpos(bnum,1), llpos(bnum,2):urpos(bnum,2)) = ...
	mult*pyrBand(pyr,pind,bnum) + (1.5-mult*range(bnum,1));
  end
    
  hh = image(d_im);
  axis('off');
  pixelAxes(size(d_im),'full');
  set(hh,'UserData',range);

end
