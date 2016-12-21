% RES = blur(IM, LEVELS, FILT)
%
% Blur an image, by filtering and downsampling LEVELS times
% (default=1), followed by upsampling and filtering LEVELS times.  The
% blurring is done with filter kernel specified by FILT (default =
% 'binom5'), which can be a string (to be passed to namedFilter), a
% vector (applied separably as a 1D convolution kernel in X and Y), or
% a matrix (applied as a 2D convolution kernel).  The downsampling is
% always by 2 in each direction.

% Eero Simoncelli, 3/04.

function res = blur(im, nlevs, filt)

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('nlevs') ~= 1) 
  nlevs = 1;
end

if (exist('filt') ~= 1) 
  filt = 'binom5';
end

%------------------------------------------------------------

res = upBlur(blurDn(im,nlevs,filt));