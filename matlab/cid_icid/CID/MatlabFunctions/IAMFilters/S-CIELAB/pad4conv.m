function newim = pad4conv(im, kernelsize, dim)
% newim = pad4conv(im, kernelsize, dim)
% 
% Pad the input image ready for convolution. The edges of the image
% (of width=kernelsize/2, or half of the image size, which ever is smaller)
%  are reflected on all sides.  
%
% kernelsize -- size of the convolution kernel in the format
%   [numRows numCol]. If one number is given, assume numRows=numCols.
% dim -- when set at 1, pad extra rows, but leave number of columns unchanged;
%        when set at 2, pad extra columns, leave number of rows unchanged;
%        when not specified, pad both columns and rows.
%
% Xuemei Zhang
% Last modified 3/1/96

if (length(kernelsize)==1)
  kernelsize = [kernelsize kernelsize];
end
if (nargin < 3)
  dim = 3;
end

[m,n] = size(im);

% If kernel is larger than image, than just pad all side with half
% the image size, otherwise pad with half the kernel size
if (kernelsize(1)>=m)
  h = floor(m/2);
else
  h = floor(kernelsize(1)/2);
end

if (kernelsize(2)>=n)
  w = floor(n/2);
else
  w = floor(kernelsize(2)/2);
end

% first reflect the upper and lower edges
if (h~=0 & dim~=2)
  im = [im; flipud(im((m-h+1):m, :))];
  im = [flipud(im(1:h, :)); im];
end

% then reflect the left and right sides
if (w~=0 & dim~=1)
  im = [im fliplr(im(:, (n-w+1):n))];
  im = [fliplr(im(:, 1:w)) im];
end

newim = im;
