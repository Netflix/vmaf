% Implementation of the improved color-image-difference metric (iCID
% metric) which predicts the perceived difference of two color images.
%
% 2014/02/27: Version 1.00
%
% For an example, see 'Example.m'.
%
% This code is supplementary material to the article:
%       J. Preiss, F. Fernandes, and P. Urban, "Color-Image Quality 
%       Assessment: From Prediction to Optimization", IEEE Transactions on 
%       Image Processing, pp. 1366-1378, Volume 23, Issue 3, March 2014
%
% Authors:  Jens Preiss, Felipe Fernandes
%           Institute of Printing Science and Technology
%           Technische Universität Darmstadt
%           preiss.science@gmail.com
%           fernandes@idd.tu-darmstadt.de
%           http://www.idd.tu-darmstadt.de/color
%           and
%           Philipp Urban
%           Fraunhofer Institute for Computer Graphics Research IGD
%           philipp.urban@igd.fraunhofer.de
%           http://www.igd.fraunhofer.de/en/Institut/Abteilungen/3DT
%
% Input:  (1) Img1: The first sRGB image being compared (string or array)
%         (2) Img2: The second sRGB image being compared (string or array)
%         (3) varargin: (optional) Set parameter for non-default usage
%
% Output: (1) Prediction: Perceived image difference between Img1 and Img2
%         (2) Maps: Image-difference maps representing image-difference
%                   features
%
% Parameters: 
%   Intent: Prediction intent of the iCID measure
%    -> 'perceptual' (default) [equal weight of lightness, chroma, and hue]
%       'hue-preserving' [stronger weight of hue]
%       'chromatic' [stronger weight of chroma and hue]
%   IAM: Image appearance model for spatial filtering
%    -> 'iCAM_YCC' (default)
%       'NONE'
%   Omit_Maps67: On biased databases chroma contrast (map 6) and chroma 
%                structure (map 7) may lead to worse prediction and should
%                then be omitted
%     -> false (default) [Maps 6 and 7 are used]
%     -> true [Maps 6 and 7 are omitted]
%   Downsampling: Automatic downsampling to set images to a better scale
%     -> true (default)
%     -> false
%
function [Prediction, Maps] = iCID(Img1, Img2, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check input arguments
if (nargin < 2)
    error('iCID:vararginChk', ['Wrong number of input arguments! ' ...
        'Number must be at least ''2'' for both input images!']);
end
if (mod(nargin, 2))
    error('iCID:vararginChk', ['Wrong number of input arguments! ' ...
        'Number must be even!']);
end

% Default parameters
Options = struct( ...
    'Intent', 'perceptual', ...
    'IAM', 'iCAM_YCC', ...
    'Omit_Maps67', false, ...
    'Downsampling', true ...
    );

% Set parameters from input
if (~isempty(varargin))
    for i = 1 : 2 : size(varargin, 2)
        if (strcmp(varargin{i}, 'Intent'))
            Options.Intent = varargin{i+1};
        elseif (strcmp(varargin{i}, 'IAM'))
            Options.IAM = varargin{i+1};
        elseif (strcmp(varargin{i}, 'Omit_Maps67'))
            Options.Omit_Maps67 = varargin{i+1};
        elseif (strcmp(varargin{i}, 'Downsampling'))
            Options.Downsampling = varargin{i+1};
        else
            warning('IDM:vararginChk', ['Wrong option ''%s''! Choose ' ...
                'between ''Intent'', ''IAM'', ''Omit_Maps67'', ' ...
                'or ''Downsampling'''], varargin{i});
        end
    end
end

% Check parameters
if (~ismember(Options.Intent, {'perceptual', 'hue-preserving', ...
        'chromatic'}))
    error('iCID:parameterChk', ['Wrong choice of prediction intent. '...
        'Choose between ''perceptual'', ''hue-preserving'', or ' ...
        '''chromatic'' prediction intent.']);
end
if (~ismember(Options.IAM, {'iCAM_YCC', 'NONE'}))
    error('iCID:parameterChk', ['Wrong image-appearance model. Choose '...
        'between ''iCAM_YCC'' or ''NONE''.']);
end
if (~isa(Options.Omit_Maps67, 'logical'))
    error('iCID:parameterChk', ['Wrong object class of variable ' ...
        '''Omit_Maps67''. Object class must be ''logical''.']);
end
if (~isa(Options.Downsampling, 'logical'))
    error('iCID:parameterChk', ['Wrong object class of variable ' ...
        '''Downsampling''. Object class must be ''logical''.']);
end

% Non-variable parameters
Options.Alpha = 3;
Options.WindowSize = 11;
Options.StdDev = 2.0;
Options.CyclesPerDegree = 20;
Options.Mode = 'valid';

% Process parameters
if (strcmp(Options.Intent, 'perceptual'))
    Options.iCID_Consts = [0.002, 10, 10, 0.002, 0.002, 10, 10];
elseif (strcmp(Options.Intent, 'hue-preserving'))
    Options.iCID_Consts = [0.002, 10, 10, 0.002, 0.02, 10, 10];
elseif (strcmp(Options.Intent, 'chromatic'))
    Options.iCID_Consts = [0.002, 10, 10, 0.02, 0.02, 10, 10];
end
if (Options.Omit_Maps67 == false)
    Options.Exponents = [1, 1, Options.Alpha, 1, 1, 1, 1];
elseif (Options.Omit_Maps67 == true)
    Options.Exponents = [1, 1, Options.Alpha, 1, 1, 0, 0];
end
Options.Window = fspecial('gaussian', Options.WindowSize, Options.StdDev);

%%%%%%%%%%%%%%% TRANSFORM IMAGES TO THE WORKING COLOR SPACE %%%%%%%%%%%%%%%
% Here we use the almost perceptually uniform and hue linear LAB2000HL
% color space.

% Get MxNx3 matrix of each image as double
if (~isa(Img1, 'numeric'))
    Img1 = imread(Img1);
end
if (~isa(Img2, 'numeric'))
    Img2 = imread(Img2);
end
if (isa(Img1, 'uint8'))
    Img1 = double(Img1) / 255;
end
if (isa(Img2, 'uint8'))
    Img2 = double(Img2) / 255;
end
if (size(Img1, 3) == 4) % For sRGB TIFF-files in MATLAB the fourth
    Img1(:,:,4) = [];   % dimension is omitted
end
if (size(Img2, 3) == 4)
    Img2(:,:,4) = [];
end

% Check image size
[M, N, D] = size(Img1);
[M2, N2, D2] = size(Img2);
if (sum([M, N, D] ~= [M2, N2, D2]))
    error('iCID:ImgSizeChk', ['Image sizes do not match! ' ...
        'The input images must have the same size!']);
end
if (D ~= 3)
    error('iCID:ImgSizeChk', ['Wrong image dimension! ' ...
        'The dimension must be ''3''!']);
end
    
% Downsample images
if (Options.Downsampling)
    f = max(1, round(min(M, N) / 256));
    if (f > 1)
        lpf = ones(f, f);
        lpf = lpf / sum(lpf(:));
        Img1 = imfilter(Img1, lpf, 'symmetric', 'same');
        Img2 = imfilter(Img2, lpf, 'symmetric', 'same');       
        Img1 = Img1(1:f:end, 1:f:end, :);
        Img2 = Img2(1:f:end, 1:f:end, :);
    end
end

% Transform images
Img1_XYZ = ImageSRGB2XYZ(Img1);
Img2_XYZ = ImageSRGB2XYZ(Img2);
switch (Options.IAM)    % Apply image appearance model on XYZ data
    case 'iCAM_YCC'
        Img1_filt    = FilterImageCSF(Img1_XYZ, ...
            'cpd', Options.CyclesPerDegree, 'wrk_space', 'YCC-RIT', ...
            'pad_image', 'both');
        Img2_filt    = FilterImageCSF(Img2_XYZ, ...
            'cpd', Options.CyclesPerDegree, 'wrk_space', 'YCC-RIT', ...
            'pad_image', 'both');
    case 'NONE'
        Img1_filt    = Img1_XYZ;
        Img2_filt    = Img2_XYZ;
end
Img1_LAB2000HL = ImageXYZ2LAB2000HL(Img1_filt);
Img2_LAB2000HL = ImageXYZ2LAB2000HL(Img2_filt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE PREMAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating the premaps is based upon the SSIM from MeTriX MuX

% Abbreviations
img1 = Img1_LAB2000HL;
img2 = Img2_LAB2000HL;
L1 = img1(:, :, 1);
A1 = img1(:, :, 2);
B1 = img1(:, :, 3);
C1_sq = A1 .^2 +  B1 .^2;
L2 = img2(:, :, 1);
A2 = img2(:, :, 2);
B2 = img2(:, :, 3);
C2_sq = A2 .^2 + B2 .^2;

C1 = sqrt(C1_sq);
C2 = sqrt(C2_sq);

% Mean intensity mu
muL1 = filter2(Options.Window, L1, Options.Mode);
muC1 = filter2(Options.Window, C1, Options.Mode);
muL2 = filter2(Options.Window, L2, Options.Mode);
muC2 = filter2(Options.Window, C2, Options.Mode);

% Standard deviation sigma
sL1_sq = filter2(Options.Window, L1 .^2, Options.Mode) - muL1 .^2;
sL1_sq(sL1_sq < 0) = 0;
sL1 = sqrt(sL1_sq);
sL2_sq = filter2(Options.Window, L2 .^2, Options.Mode) - muL2 .^2;
sL2_sq(sL2_sq < 0) = 0;
sL2 = sqrt(sL2_sq);

sC1_sq = filter2(Options.Window, C1_sq, Options.Mode) - muC1 .^2;
sC1_sq(sC1_sq < 0) = 0;
sC1 = sqrt(sC1_sq);
sC2_sq = filter2(Options.Window, C2_sq, Options.Mode) - muC2 .^2;
sC2_sq(sC2_sq < 0) = 0;
sC2 = sqrt(sC2_sq);

% Get mixed terms (dL_sq, dC_sq, dH_sq, sL12)
dL_sq = (muL1 - muL2) .^2;
dC_sq = (muC1 - muC2) .^2;
dH_sq = filter2(Options.Window, sqrt((A1 - A2) .^2 + (B1 - B2) .^2 - ...
    (sqrt(C1_sq) - sqrt(C2_sq)) .^2), Options.Mode) .^2;
sL12 = filter2(Options.Window, L1 .* L2, Options.Mode) - muL1 .* muL2;
sC12 = filter2(Options.Window, C1 .* C2, Options.Mode) - muC1 .* muC2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE MAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Parameters
Maps_inv = cell(1, 6);
Maps = cell(1, 6);

% 1) Lightness difference
Maps_inv{1} = 1 ./ (Options.iCID_Consts(1) * dL_sq + 1);
Maps{1} = real(1 - Maps_inv{1});

% 2) Lightness contrast
Maps_inv{2} = (Options.iCID_Consts(2) + 2 * sL1 .* sL2) ./ ...
              (Options.iCID_Consts(2) + sL1_sq + sL2_sq);
Maps{2} = real(1 - Maps_inv{2});

% 3) Lightness structure
Maps_inv{3} = (Options.iCID_Consts(3) + abs(sL12)) ./ ...
    (Options.iCID_Consts(3) + sL1 .* sL2);
Maps{3} = real(1 - Maps_inv{3});

% 4) Chroma difference
Maps_inv{4} = 1 ./ (Options.iCID_Consts(4) * dC_sq + 1);
Maps{4} = real(1 - Maps_inv{4});

% 5) Hue difference
Maps_inv{5} = 1 ./ (Options.iCID_Consts(5) * dH_sq + 1);
Maps{5} = real(1 - Maps_inv{5});

% 6) Chroma contrast 
Maps_inv{6} = (Options.iCID_Consts(6) + 2 * sC1 .* sC2) ./ ...
              (Options.iCID_Consts(6) + sC1_sq + sC2_sq);
Maps{6} = real(1 - Maps_inv{6});

% 3) Chroma structure 
Maps_inv{7} = (Options.iCID_Consts(7) + abs(sC12)) ./ ...
    (Options.iCID_Consts(7) + sC1 .* sC2);
Maps{7} = real(1 - Maps_inv{7});

%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Potentiate maps with exponents
for i = 1 : 7
    Maps_inv{i} = Maps_inv{i} .^ Options.Exponents(i);
end

% Compute prediction pixelwise
Prediction = 1 - mean2(Maps_inv{1} .* Maps_inv{2} .* ...
    Maps_inv{3} .* Maps_inv{4} .* Maps_inv{5} .* ...
    Maps_inv{6} .* Maps_inv{7});

% Occasionally, the prediction has a very small imaginary part; we keep
% only the real part of the prediction
Prediction = real(Prediction);

end