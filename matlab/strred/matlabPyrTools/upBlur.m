% RES = upBlur(IM, LEVELS, FILT)
%
% Upsample and blur an image.  The blurring is done with filter
% kernel specified by FILT (default = 'binom5'), which can be a string
% (to be passed to namedFilter), a vector (applied separably as a 1D
% convolution kernel in X and Y), or a matrix (applied as a 2D
% convolution kernel).  The downsampling is always by 2 in each
% direction.
%
% The procedure is applied recursively LEVELS times (default=1).

% Eero Simoncelli, 4/97.

function res = upBlur(im, nlevs, filt)

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

if nlevs > 1
  im = upBlur(im,nlevs-1,filt);
end

if (nlevs >= 1)
  if (any(size(im)==1))
    if (size(im,1)==1)
      filt = filt';
    end
    res = upConv(im,filt,'reflect1',(size(im)~=1)+1);
  elseif (any(size(filt)==1))
    filt = filt(:);
    res = upConv(im,filt,'reflect1',[2 1]);
    res = upConv(res,filt','reflect1',[1 2]);
  else
    res = upConv(im,filt,'reflect1',[2 2]);
  end
else
  res = im;
end
