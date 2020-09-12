function outImage = changeColorSpace(inImage,colorMatrix)
% outImage = changeColorSpace(inImage,colorMatrix)
%
% The input image consists of three input images, say R,G,B, joined as 
%
%		inImage = [ R G B];
%
% The output image has the same format
%
% The 3 x 3 color matrix converts column vectors in the input image
% representation into column vectors in the output representation.
%
% Xuemei Zhang, Brian Wandell  03/08/96
% Last Modified  4/15/98

insize = size(inImage);

% We put the pixels in the input image into the rows of a very
% large matrix
%
inImage = reshape(inImage, prod(insize)/3, 3);

% We post-multiply by colorMatrix' to convert the pixels to the output 
% color space
%
outImage = inImage*colorMatrix';

% Now we put the output image in the basic shape we use
%
outImage = reshape(outImage, insize);
