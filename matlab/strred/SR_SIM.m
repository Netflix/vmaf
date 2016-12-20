function sim = SR_SIM(image1, image2)
% ========================================================================
% SR_SIM Index with automatic downsampling, Version 1.0
% Copyright(c) 2011 Lin ZHANG
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is hereQ
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for calculating the
% Spectral Residual based Similarity (SR-SIM) index between two images. For
% more details, please refer to our paper:
% Lin Zhang and Hongyu Li, "SR-SIM: A fast and high performance IQA index based on spectral residual", in: Proc. ICIP 2012.
%
%----------------------------------------------------------------------
%
%Input : (1) image1: the first image being compared
%        (2) image2: the second image being compared
%
%Output: sim: the similarity score between two images, a real number
%        
%-----------------------------------------------------------------------
[rows, cols, junk] = size(image1);
if junk == 3
    Y1 = 0.299 * double(image1(:,:,1)) + 0.587 * double(image1(:,:,2)) + 0.114 * double(image1(:,:,3));
    Y2 = 0.299 * double(image2(:,:,1)) + 0.587 * double(image2(:,:,2)) + 0.114 * double(image2(:,:,3));
else
    Y1 = double(image1);
    Y2 = double(image2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%
% Download the image
%%%%%%%%%%%%%%%%%%%%%%%%%
minDimension = min(rows,cols);
F = max(1,round(minDimension / 256));
aveKernel = fspecial('average',F);

aveY1 = conv2(Y1, aveKernel,'same');
aveY2 = conv2(Y2, aveKernel,'same');
Y1 = aveY1(1:F:rows,1:F:cols);
Y2 = aveY2(1:F:rows,1:F:cols);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the visual saliency maps
%%%%%%%%%%%%%%%%%%%%%%%%%
saliencyMap1 = spectralResidueSaliency(Y1);
saliencyMap2 = spectralResidueSaliency(Y2);
%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the gradient map
%%%%%%%%%%%%%%%%%%%%%%%%%
dx = [3 0 -3; 10 0 -10;  3  0 -3]/16;
dy = [3 10 3; 0  0   0; -3 -10 -3]/16;
IxY1 = conv2(Y1, dx, 'same');     
IyY1 = conv2(Y1, dy, 'same');    
gradientMap1 = sqrt(IxY1.^2 + IyY1.^2);

IxY2 = conv2(Y2, dx, 'same');     
IyY2 = conv2(Y2, dy, 'same');    
gradientMap2 = sqrt(IxY2.^2 + IyY2.^2);

%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculate the SR-SIM
%%%%%%%%%%%%%%%%%%%%%%%%%
C1 = 0.40; %fixed
C2 = 225; 
alpha = 0.50;%fixed

GBVSSimMatrix = (2 * saliencyMap1 .* saliencyMap2 + C1) ./ (saliencyMap1.^2 + saliencyMap2.^2 + C1);
gradientSimMatrix = (2*gradientMap1.*gradientMap2 + C2) ./(gradientMap1.^2 + gradientMap2.^2 + C2);

weight = max(saliencyMap1, saliencyMap2);
SimMatrix = GBVSSimMatrix .* (gradientSimMatrix .^ alpha) .* weight;
sim = sum(sum(SimMatrix)) / sum(weight(:));

return;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function saliencyMap = spectralResidueSaliency(image)
%this function is used to calculate the visual saliency map for the given
%image using the spectral residue method proposed by Xiaodi Hou and Liqing
%Zhang. For more details about this method, you can refer to the paper:
%Saliency detection: a spectral residual approach.

%there are some parameters needed to be adjusted
scale = 0.25; %fixed
aveKernelSize = 3; %fixed
gauSigma = 3.8; %fixed
gauSize = 10; %fixed

inImg = imresize(image, scale);

%%%% Spectral Residual
myFFT = fft2(inImg);
myLogAmplitude = log(abs(myFFT));
myPhase = angle(myFFT);

mySpectralResidual = myLogAmplitude - imfilter(myLogAmplitude, fspecial('average', aveKernelSize), 'replicate');
saliencyMap = abs(ifft2(exp(mySpectralResidual + 1i*myPhase))).^2;

%%%% After Effect
saliencyMap = mat2gray(imfilter(saliencyMap, fspecial('gaussian', [gauSize, gauSize], gauSigma)));
saliencyMap = imresize(saliencyMap,[size(image,1) size(image,2)]);