% IM = mkGaussian(SIZE, COVARIANCE, MEAN, AMPLITUDE)
% 
% Compute a matrix with dimensions SIZE (a [Y X] 2-vector, or a
% scalar) containing a Gaussian function, centered at pixel position
% specified by MEAN (default = (size+1)/2), with given COVARIANCE (can
% be a scalar, 2-vector, or 2x2 matrix.  Default = (min(size)/6)^2),
% and AMPLITUDE.  AMPLITUDE='norm' (default) will produce a
% probability-normalized function.  All but the first argument are
% optional.

% Eero Simoncelli, 6/96.

function [res] = mkGaussian(sz, cov, mn, ampl)

sz = sz(:);
if (size(sz,1) == 1)
  sz = [sz,sz];
end

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('cov') ~= 1)
  cov = (min(sz(1),sz(2))/6)^2;
end

if (exist('mn') ~= 1)
  mn = (sz+1)/2;
end

if (exist('ampl') ~= 1)
  ampl = 'norm';
end

%------------------------------------------------------------

[xramp,yramp] = meshgrid([1:sz(2)]-mn(2),[1:sz(1)]-mn(1));

if (sum(size(cov)) == 2)  % scalar
  if (strcmp(ampl,'norm'))  
    ampl = 1/(2*pi*cov(1));
  end
  e = (xramp.^2 + yramp.^2)/(-2 * cov);
elseif (sum(size(cov)) == 3) % a 2-vector
  if (strcmp(ampl,'norm'))  
    ampl = 1/(2*pi*sqrt(cov(1)*cov(2)));
  end
  e = xramp.^2/(-2 * cov(2)) + yramp.^2/(-2 * cov(1));
else
  if (strcmp(ampl,'norm'))  
    ampl = 1/(2*pi*sqrt(det(cov)));
  end
  cov = -inv(cov)/2;
  e = cov(2,2)*xramp.^2 + (cov(1,2)+cov(2,1))*(xramp.*yramp) ...
      + cov(1,1)*yramp.^2;
end
  
res = ampl .* exp(e);
