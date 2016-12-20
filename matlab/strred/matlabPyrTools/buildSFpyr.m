% [PYR, INDICES, STEERMTX, HARMONICS] = buildSFpyr(IM, HEIGHT, ORDER, TWIDTH)
%
% Construct a steerable pyramid on matrix IM, in the Fourier domain.
% This is similar to buildSpyr, except that:
%
%    + Reconstruction is exact (within floating point errors)
%    + It can produce any number of orientation bands.
%    - Typically slower, especially for non-power-of-two sizes.
%    - Boundary-handling is circular.
%
% HEIGHT (optional) specifies the number of pyramid levels to build. Default
% is maxPyrHt(size(IM),size(FILT));
%
% The squared radial functions tile the Fourier plane, with a raised-cosine
% falloff.  Angular functions are cos(theta-k\pi/(K+1))^K, where K is
% the ORDER (one less than the number of orientation bands, default= 3).
%
% TWIDTH is the width of the transition region of the radial lowpass
% function, in octaves (default = 1, which gives a raised cosine for
% the bandpass filters).
%
% PYR is a vector containing the N pyramid subbands, ordered from fine
% to coarse.  INDICES is an Nx2 matrix containing the sizes of
% each subband.  This is compatible with the MatLab Wavelet toolbox.
% See the function STEER for a description of STEERMTX and HARMONICS.

% Eero Simoncelli, 5/97.
% See http://www.cns.nyu.edu/~eero/STEERPYR/ for more
% information about the Steerable Pyramid image decomposition.

function [pyr,pind,steermtx,harmonics] = buildSFpyr(im, ht, order, twidth)

%-----------------------------------------------------------------
%% DEFAULTS:

max_ht = floor(log2(min(size(im)))) - 2;

if (exist('ht') ~= 1)
  ht = max_ht;
else
  if (ht > max_ht)
    error(sprintf('Cannot build pyramid higher than %d levels.',max_ht));
  end
end

if (exist('order') ~= 1)
  order = 3;
elseif ((order > 15)  | (order < 0))
  fprintf(1,'Warning: ORDER must be an integer in the range [0,15]. Truncating.\n');
  order = min(max(order,0),15);
else
  order = round(order);
end
nbands = order+1;

if (exist('twidth') ~= 1)
  twidth = 1;
elseif (twidth <= 0)
  fprintf(1,'Warning: TWIDTH must be positive.  Setting to 1.\n');
  twidth = 1;
end

%-----------------------------------------------------------------
%% Steering stuff:

if (mod((nbands),2) == 0)
  harmonics = [0:(nbands/2)-1]'*2 + 1;
else
  harmonics = [0:(nbands-1)/2]'*2;
end

steermtx = steer2HarmMtx(harmonics, pi*[0:nbands-1]/nbands, 'even');

%-----------------------------------------------------------------

dims = size(im);
ctr = ceil((dims+0.5)/2);

[xramp,yramp] = meshgrid( ([1:dims(2)]-ctr(2))./(dims(2)/2), ...
    ([1:dims(1)]-ctr(1))./(dims(1)/2) );
angle = atan2(yramp,xramp);
log_rad = sqrt(xramp.^2 + yramp.^2);
log_rad(ctr(1),ctr(2)) =  log_rad(ctr(1),ctr(2)-1);
log_rad  = log2(log_rad);

%% Radial transition function (a raised cosine in log-frequency):
[Xrcos,Yrcos] = rcosFn(twidth,(-twidth/2),[0 1]);
Yrcos = sqrt(Yrcos);

YIrcos = sqrt(1.0 - Yrcos.^2);
lo0mask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
imdft = fftshift(fft2(im));
lo0dft =  imdft .* lo0mask;

[pyr,pind] = buildSFpyrLevs(lo0dft, log_rad, Xrcos, Yrcos, angle, ht, nbands);

hi0mask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);
hi0dft =  imdft .* hi0mask;
hi0 = ifft2(ifftshift(hi0dft));

pyr = [real(hi0(:)) ; pyr];
pind = [size(hi0); pind];
