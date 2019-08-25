% RESDFT = reconSFpyrLevs(PYR,INDICES,LOGRAD,XRCOS,YRCOS,ANGLE,NBANDS,LEVS,BANDS)
%
% Recursive function for reconstructing levels of a steerable pyramid
% representation.  This is called by reconSFpyr, and is not usually
% called directly.

% Eero Simoncelli, 5/97.

function resdft = reconSFpyrLevs(pyr,pind,log_rad,Xrcos,Yrcos,angle,nbands,levs,bands);

lo_ind = nbands+1;
dims = pind(1,:);
ctr = ceil((dims+0.5)/2);

%  log_rad = log_rad + 1;
Xrcos = Xrcos - log2(2);  % shift origin of lut by 1 octave.

if any(levs > 1)

  lodims = ceil((dims-0.5)/2);
  loctr = ceil((lodims+0.5)/2);
  lostart = ctr-loctr+1;
  loend = lostart+lodims-1;
  nlog_rad = log_rad(lostart(1):loend(1),lostart(2):loend(2));
  nangle = angle(lostart(1):loend(1),lostart(2):loend(2));

  if  (size(pind,1) > lo_ind)
    nresdft = reconSFpyrLevs( pyr(1+sum(prod(pind(1:lo_ind-1,:)')):size(pyr,1)),...
	pind(lo_ind:size(pind,1),:), ...
	nlog_rad, Xrcos, Yrcos, nangle, nbands,levs-1, bands);
  else
    nresdft = fftshift(fft2(pyrBand(pyr,pind,lo_ind)));
  end

  YIrcos = sqrt(abs(1.0 - Yrcos.^2));
  lomask = pointOp(nlog_rad, YIrcos, Xrcos(1), Xrcos(2)-Xrcos(1), 0);

  resdft = zeros(dims);
  resdft(lostart(1):loend(1),lostart(2):loend(2)) = nresdft .* lomask;

else

  resdft = zeros(dims);

end

	
if any(levs == 1)

  lutsize = 1024;
  Xcosn = pi*[-(2*lutsize+1):(lutsize+1)]/lutsize;  % [-2*pi:pi]
  order = nbands-1;
  %% divide by sqrt(sum_(n=0)^(N-1)  cos(pi*n/N)^(2(N-1)) )
  const = (2^(2*order))*(factorial(order)^2)/(nbands*factorial(2*order));
  Ycosn = sqrt(const) * (cos(Xcosn)).^order;
  himask = pointOp(log_rad, Yrcos, Xrcos(1), Xrcos(2)-Xrcos(1),0);

  ind = 1;
  for b = 1:nbands
    if any(bands == b)
      anglemask = pointOp(angle,Ycosn,Xcosn(1)+pi*(b-1)/nbands,Xcosn(2)-Xcosn(1));
      band = reshape(pyr(ind:ind+prod(dims)-1), dims(1), dims(2));
      banddft = fftshift(fft2(band));
      resdft = resdft + (sqrt(-1))^(nbands-1) * banddft.*anglemask.*himask;
    end
    ind = ind + prod(dims);
  end
end

