% [PYR, INDICES] = buildSFpyrLevs(LODFT, LOGRAD, XRCOS, YRCOS, ANGLE, HEIGHT, NBANDS)
%
% Recursive function for constructing levels of a steerable pyramid.  This
% is called by buildSFpyr, and is not usually called directly.

% Eero Simoncelli, 5/97.

function [pyr,pind] = buildSFpyrLevs(lodft,log_rad,Xrcos,Yrcos,angle,ht,nbands);

if (ht <= 0)

  lo0 = ifft2(ifftshift(lodft));
  pyr = real(lo0(:));
  pind = size(lo0);

else

  bands = zeros(prod(size(lodft)), nbands);
  bind = zeros(nbands,2);

%  log_rad = log_rad + 1;
  Xrcos = Xrcos - log2(2);  % shift origin of lut by 1 octave.

  lutsize = 1024;
  Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;  % [-2*pi:pi]
  order = nbands-1;
  %% divide by sqrt(sum_(n=0)^(N-1)  cos(pi*n/N)^(2(N-1)) )
  %% Thanks to Patrick Teo for writing this out :)
  const = (2^(2*order))*(factorial(order)^2)/(nbands*factorial(2*order));
  Ycosn = sqrt(const) * (cos(Xcosn)).^order;
  himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);

  for b = 1:nbands
    anglemask = pointOp(angle, Ycosn, Xcosn(1)+pi*(b-1)/nbands, Xcosn(2)-Xcosn(1));
    banddft = ((-sqrt(-1))^(nbands-1)) .* lodft .* anglemask .* himask;
    band = ifft2(ifftshift(banddft));

    bands(:,b) = real(band(:));
    bind(b,:)  = size(band);
  end

  dims = size(lodft);
  ctr = ceil((dims+0.5)/2);
  lodims = ceil((dims-0.5)/2);
  loctr = ceil((lodims+0.5)/2);
  lostart = ctr-loctr+1;
  loend = lostart+lodims-1;

  log_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
  angle = angle(lostart(1):loend(1),lostart(2):loend(2));
  lodft = lodft(lostart(1):loend(1),lostart(2):loend(2));
  YIrcos = abs(sqrt(1.0 - Yrcos.^2));
  lomask = pointOp(log_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);

  lodft = lomask .* lodft;

  [npyr,nind] = buildSFpyrLevs(lodft, log_rad, Xrcos, Yrcos, angle, ht-1, nbands);

  pyr = [bands(:); npyr];
  pind = [bind; nind];

end

