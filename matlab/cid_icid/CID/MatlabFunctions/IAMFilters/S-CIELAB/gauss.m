function gauss = gauss(halfWidth, width)
% gauss(halfWidth, width)
% 	Returns a 1D Gaussian vector.  The gaussian sums to one.  The
%	halfWidth must be greater than one.
% 
% The halfwidth specifies the width of the gaussian between the points
% where it obtains half of its maximum value.  The width indicates the
% gaussians width in pixels.
%
% Rick Anthony
% 8/24/93

if (nargin < 2)
    error('Two input arguments required');
end

alpha = 2*sqrt(log(2))/(halfWidth-1);
x = (1:width)-round(width/2);

gauss = exp(-alpha*alpha*x.*x);
gauss = gauss/sum(sum(gauss));
