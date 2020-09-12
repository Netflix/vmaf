function [k1, k2, k3] = separableFilters(sampPerDeg, dimension)
%
% [k1, k2, k3] = separableFilters(sampPerDeg, dimension)
%
% Create the pattern-color separable filters according to
% the Poirson & Wandell 1993 fitted spatial response. The filters
% are each weighted sum of 2 or 3 gaussians.
%
% sampPerDeg -- filter resolution in samples per degree of visual angle.
%
% dimension -- specifies whether the created filters should be 1-D or 2-D.
%
%   dimension = 1: generate the linespread of the filters;
%	This is useful for 1-d image calculations, say for theoretical 
%	work.
%
%   dimension = 2: generate the pointspread of the filters;
%	This is useful if you just want to create an image of the filters
%
%   dimension = 3: generate the pointspread in the form that can be used by 
% 	separableConv.  The result is a set of 1-d filters that can be applied
%	to the rows and cols of the image (separably).
%
% Functions called: sumGauss.
%
% Xuemei Zhang  1/28/96
% Last Modified 2/29/95

if (nargin==1)
  dimension = 1;
end

minSAMPPERDEG = 224;
% if sampPerDeg is smaller than minSAMPPERDEG, need to upsample image data before
% doing the filtering. This can be done equivalently by convolving
% the filters with the upsampling matrix, then downsample it.

if ((sampPerDeg<minSAMPPERDEG) & dimension==2)
  disp(['sampPerDeg should be greater than or equal to' num2str(minSAMPPERDEG) '.']);
end

if ( (sampPerDeg<minSAMPPERDEG) & dimension~=2 )
  uprate = ceil(minSAMPPERDEG/sampPerDeg);
  sampPerDeg = sampPerDeg * uprate;
else
  uprate = 1;
end


% These are the parameters for generating the filters,
% expressed as weighted sum of two or three gaussians, 
% in the format [halfwidth weight halfwidth weight ...].
% The halfwidths are in degrees of visual angle.
%x1 = [0.05      0.9207    0.225    0.105    7.0   -0.1080];
%x2 = [0.0685    0.5310    0.826    0.33];
%x3 = [0.0920    0.4877    0.6451    0.3711];
%% these are the same filter parameters, except that the weights
%% are normalized to sum to 1 -- this eliminates the need to
%% normalize after the filters are generated
x1 = [0.05     1.00327  0.225  0.114416  7.0  -0.117686];
x2 = [0.0685    0.616725    0.826    0.383275];
x3 = [0.0920    0.567885    0.6451    0.432115];

% Convert the unit of halfwidths from visual angle to pixels.
x1([1 3 5]) = x1([1 3 5]) * sampPerDeg;
x2([1 3]) = x2([1 3]) * sampPerDeg;
x3([1 3]) = x3([1 3]) * sampPerDeg;

% Limit the width of filters to 1 degree visual angle, and 
% odd number of sampling points (so that the gaussians generated 
% from Rick's gauss routine are symmetric).
width = ceil(sampPerDeg/2) * 2 - 1;

% Generate the filters
if (dimension < 3)
  k1 = sumGauss([width x1], dimension);
  k2 = sumGauss([width x2], dimension);
  k3 = sumGauss([width x3], dimension);

  % make sure the filters sum to 1
%  k1 = k1/sum(k1(:));
%  k2 = k2/sum(k2(:));
%  k3 = k3/sum(k3(:));

else

% In this case, we do not compute the filter. We compute the
% individual Gaussians rather than the sums of Gaussians.  These Gaussians
% are used in the row and col separable convolutions.
%
%  k1 = sumGauss([width x1], 2);
%  k2 = sumGauss([width x2], 2);
%  k3 = sumGauss([width x3], 2);
%  s1 = (sum(k1(:)));
%  s2 = (sum(k2(:)));
%  s3 = (sum(k3(:)));

% k1 contains the three 1-d kernels that are used by the light/dark
% system
%
%  k1 = [gauss(x1(1), width) * sqrt(abs(x1(2))/s1) * sign(x1(2)); ...
%        gauss(x1(3), width) * sqrt(abs(x1(4))/s1) * sign(x1(4)); ...
%        gauss(x1(5), width) * sqrt(abs(x1(6))/s1) * sign(x1(6))];
  k1 = [gauss(x1(1), width) * sqrt(abs(x1(2))) * sign(x1(2)); ...
        gauss(x1(3), width) * sqrt(abs(x1(4))) * sign(x1(4)); ...
        gauss(x1(5), width) * sqrt(abs(x1(6))) * sign(x1(6))];

% These are the two 1-d kernels used by red/green
%
%  k2 = [gauss(x2(1), width) * sqrt(abs(x2(2))/s2) * sign(x2(2)); ...
%        gauss(x2(3), width) * sqrt(abs(x2(4))/s2) * sign(x2(4))];
  k2 = [gauss(x2(1), width) * sqrt(abs(x2(2))) * sign(x2(2)); ...
        gauss(x2(3), width) * sqrt(abs(x2(4))) * sign(x2(4))];

% These are the two 1-d kernels used by blue/yellow
%
%  k3 = [gauss(x3(1), width) * sqrt(abs(x3(2))/s3) * sign(x3(2)); ...
%        gauss(x3(3), width) * sqrt(abs(x3(4))/s3) * sign(x3(4))];
  k3 = [gauss(x3(1), width) * sqrt(abs(x3(2))) * sign(x3(2)); ...
        gauss(x3(3), width) * sqrt(abs(x3(4))) * sign(x3(4))];
end

%% upsample and downsample 
%
% More explanation
%
if ( (dimension~=2) & uprate>1 )
  upcol = [1:uprate (uprate-1):(-1):1]/uprate;
  s = length(upcol);
  upcol = resize(upcol, [1 s+width-1]);
  up1 = conv2(k1, upcol, 'same');
  up2 = conv2(k2, upcol, 'same');
  up3 = conv2(k3, upcol, 'same');
  s = size(up1, 2);
  mid = ceil(s/2);
  downs = [fliplr([mid:(-uprate):1]) (mid+uprate):uprate:size(up1,2)];
  k1 = up1(:, downs);
  k2 = up2(:, downs);
  k3 = up3(:, downs);
end

