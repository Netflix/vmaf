function result = separableConv(im, xkernels, ykernels)
%
% Two-dimensional convolution with X-Y separable kernels.
% 
% im is the input matric. im is reflected on all sides before
%   convolution.
% xkernels and ykernels are both row vectors. If ykernels is not
%   given, then use xkernels as ykernels.
% If xkernels and ykernels are matrices, each row is taken as
%   one convolution kernel and convolved with the image, and the
%   sum of the results is returned.
% 
% This function is useful when a 2-D filter can be represented by
% a linear combination of several separable filters. If the number
% of separable filters that form the 2-D filter is not too large,
% this can be substantially faster than conv2.
%
% From (put citation here)
%
% Xuemei Zhang 3/1/96

if (nargin < 3)
  ykernels = xkernels;
end

imsize = size(im);
w1 = pad4conv(im, size(xkernels,2), 2);

result = 0;
for j=1:size(xkernels,1)
  % first convovle in the horizontal direction
  p = conv2(w1, xkernels(j,:));
  p = resize(p, imsize);

  % then the vertical direction
  w2 = pad4conv(p, size(ykernels,2), 1);
  p = conv2(w2, (ykernels(j,:))');
  p = resize(p, imsize);

  % result is sum of several separable convolutions
  result = result + p;
end
