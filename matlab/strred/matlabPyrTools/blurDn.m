% RES = blurDn(IM, LEVELS, FILT)
%
% Blur and downsample an image.  The blurring is done with filter
% kernel specified by FILT (default = 'binom5'), which can be a string
% (to be passed to namedFilter), a vector (applied separably as a 1D
% convolution kernel in X and Y), or a matrix (applied as a 2D
% convolution kernel).  The downsampling is always by 2 in each
% direction.
%
% The procedure is applied recursively LEVELS times (default=1).

% Eero Simoncelli, 3/97.

function res = blurDn(im, nlevs, filt)

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('nlevs') ~= 1) 
  nlevs = 1;
end

if (exist('filt') ~= 1) 
  filt = 'binom5';
end

%------------------------------------------------------------

if isstr(filt)
  filt = namedFilter(filt);
end  

filt = filt/sum(filt(:));

if nlevs > 1
  im = blurDn(im,nlevs-1,filt);
end

if (nlevs >= 1)
  if (any(size(im)==1))
    if (~any(size(filt)==1))
      error('Cant  apply 2D filter to 1D signal');
    end
    if (size(im,2)==1)
      filt = filt(:);
    else
      filt = filt(:)';
    end
    res = corrDn(im,filt,'reflect1',(size(im)~=1)+1);
  elseif (any(size(filt)==1))
    filt = filt(:);
    res = corrDn(im,filt,'reflect1',[2 1]);
    res = corrDn(res,filt','reflect1',[1 2]);
  else
    res = corrDn(im,filt,'reflect1',[2 2]);
  end
else
  res = im;
end
