% Implementation of the color-image-difference measure (CID measure),
% which predicts the perceived difference of two images.
%
% 2013/05/31: Version 1.0 
%
% If you wish to set the parameters manually, see 'CID_advanced.m'. 
% For an example, see 'Example.m'.
%
% This code is supplementary material to the article:
%           I. Lissner, J. Preiss, P. Urban, M. Scheller Lichtenauer, and 
%           P. Zolliker, "Image-Difference Prediction: From Grayscale to
%           Color", IEEE Transactions on Image Processing, Vol. 22, 
%           Issue 2, pp. 435-446 (2013).
%
% Authors:  Ingmar Lissner, Jens Preiss, Philipp Urban
%           Institute of Printing Science and Technology
%           Technische Universität Darmstadt
%           {lissner,preiss,urban}@idd.tu-darmstadt.de
%           http://www.idd.tu-darmstadt.de/color
%
%           Matthias Scheller Lichtenauer, Peter Zolliker
%           Empa, Swiss Federal Laboratories for
%                 Materials Science and Technology
%           Laboratory for Media Technology
%           {matthias.scheller,peter.zolliker}@empa.ch
%           http://empamedia.ethz.ch
%
% Input:  (1) Img1: The first sRGB image being compared (string or array)
%         (2) Img2: The second sRGB image being compared (string or array)
%
% Output: (1) Prediction: Perceived image difference between Img1 and Img2
%         (2) Maps: Image-difference maps representing image-difference
%                   features
%
% In this implementation, the default parameters proposed in the paper are
% used:   - IDF constants: [0.002, 0.1, 0.1, 0.002, 0.008]
%         - Image appearance model: 'S-CIELAB'
%         - Cycles per degree: 20
%         - Single scale
%         - Factorial combination model
%         - Map-wise combination of the IDFs
function [Prediction, Maps] = CID(Img1, Img2)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check input arguments
if (nargin ~= 2)
    error('IDM:vararginChk', ['Wrong number of input arguments! ' ...
        'Number must be exactly ''2'' for both input images!']);
end

%%%%%%%%%%%%%%% TRANSFORM IMAGES TO THE WORKING COLOR SPACE %%%%%%%%%%%%%%%
% Here we use the almost perceptually uniform color space LAB2000HL.

% Get MxNx3 matrix of each image
if (~isa(Img1, 'numeric'))
    Img1 = imread(Img1);
end
if (~isa(Img2, 'numeric'))
    Img2 = imread(Img2);
end
if (size(Img1, 3) == 4) % For TIFF files in Matlab the fourth dimension is
    Img1(:,:,4) = [];   % omitted
end
if (size(Img2, 3) == 4)
    Img2(:,:,4) = [];
end

% Check image size
if (size(Img1, 3) ~= 3)
    error('IDM:ImgSizeChk', ['Wrong image dimension! ' ...
        'The dimension must be ''3''!']);
end
if (sum(size(Img1) ~= size(Img2)))
    error('IDM:ImgSizeChk', ['Image sizes do not match! ' ...
        'The input images must have the same size!']);
end

% Transform image 1 and 2
CyclesPerDegree = 20;
Img1_XYZ = ImageSRGB2XYZ(Img1);
Img1_filt    = scielab_simple(2 * CyclesPerDegree, Img1_XYZ);
Img1_LAB2000HL = ImageXYZ2LAB2000HL(Img1_filt);
Img2_XYZ = ImageSRGB2XYZ(Img2);
Img2_filt    = scielab_simple(2 * CyclesPerDegree, Img2_XYZ);
Img2_LAB2000HL = ImageXYZ2LAB2000HL(Img2_filt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE PREMAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating the premaps is based upon the SSIM from MeTriX MuX

% Parameters
Window = fspecial('gaussian', 11, 1.5);

% Abbreviations
img1 = Img1_LAB2000HL;
img2 = Img2_LAB2000HL;
L1 = img1(:, :, 1);
A1 = img1(:, :, 2);
B1 = img1(:, :, 3);
Chr1_sq = A1 .^2 +  B1 .^2;
L2 = img2(:, :, 1);
A2 = img2(:, :, 2);
B2 = img2(:, :, 3);
Chr2_sq = A2 .^2 + B2 .^2;

% Mean intensity mu
muL1 = filter2(Window, L1, 'valid');
muC1 = filter2(Window, sqrt(Chr1_sq), 'valid');
muL2 = filter2(Window, L2, 'valid');
muC2 = filter2(Window, sqrt(Chr2_sq), 'valid');

% Standard deviation sigma
sL1_sq = filter2(Window, L1 .^2, 'valid') - muL1 .^2;
sL1_sq(sL1_sq < 0) = 0;
sL1 = sqrt(sL1_sq);
sL2_sq = filter2(Window, L2 .^2, 'valid') - muL2 .^2;
sL2_sq(sL2_sq < 0) = 0;
sL2 = sqrt(sL2_sq);

% Get mixed terms (dL_sq, dC_sq, dH_sq, sL12)
dL_sq = (muL1 - muL2) .^2;
dC_sq = (muC1 - muC2) .^2;
dH_sq = filter2(Window, sqrt((A1 - A2) .^2 + (B1 - B2) .^2 - ...
    (sqrt(Chr1_sq) - sqrt(Chr2_sq)) .^2), 'valid') .^2;
sL12 = filter2(Window, L1 .* L2, 'valid') - muL1 .* muL2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE MAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters
IDFConsts = [0.002, 0.1, 0.1, 0.002, 0.008];
Maps_inv = cell(1, 5);
Maps = cell(1, 5);

% IDF 1) Lightness comparison
Maps_inv{1} = 1 ./ (IDFConsts(1) * dL_sq + 1);
Maps{1} = real(1 - Maps_inv{1});

% 2) Lightness-contrast comparison
Maps_inv{2} = (IDFConsts(2) + 2 * sL1 .* sL2) ./ ...
              (IDFConsts(2) + sL1_sq + sL2_sq);
Maps{2} = real(1 - Maps_inv{2});

% 3) Lightness-structure comparison
Maps_inv{3} = (IDFConsts(3) + sL12) ./ (IDFConsts(3) + sL1 .* sL2);
Maps{3} = real(1 - Maps_inv{3});

% 4) Chroma comparison
Maps_inv{4} = 1 ./ (IDFConsts(4) * dC_sq + 1);
Maps{4} = real(1 - Maps_inv{4});

% 5) Hue comparison
Maps_inv{5} = 1 ./ (IDFConsts(5) * dH_sq + 1);
Maps{5} = real(1 - Maps_inv{5});

%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Compute mean before combining
[M, N] = size(Maps_inv{1});
NumEl = M * N;
IDF1 = sum(Maps_inv{1}(:)) / NumEl; % Faster than mean2 if used
IDF2 = sum(Maps_inv{2}(:)) / NumEl; %     more than once
IDF3 = sum(Maps_inv{3}(:)) / NumEl; %
IDF4 = sum(Maps_inv{4}(:)) / NumEl; %
IDF5 = sum(Maps_inv{5}(:)) / NumEl; %

% Compute prediction by combining the means
Prediction = 1 - IDF1 * IDF2 * IDF3 * IDF4 * IDF5;

% Occasionally, the prediction has a very small imaginary part; we keep
% only the real part of the prediction
Prediction = real(Prediction);

end