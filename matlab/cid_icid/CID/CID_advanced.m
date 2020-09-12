% Implementation of the color-image-difference measure (CID measure),
% which predicts the perceived difference of two images.
%
% 2013/05/31: Version 1.0 
%
% For an easy-to-use implementation with default parameters, see 'CID.m'.
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
%         (3) varargin: (optional) Set parameter for non-default usage
%
% Output: (1) Prediction: Perceived image difference between Img1 and Img2
%         (2) Maps: Image-difference maps representing image-difference
%                   features
%
% Parameter: IDFChoice: Which IDFs to use
%            	-> Combination of integers between 1 and 6 with
%                  IDF1 ^= Lightness comparison,
%                  IDF2 ^= Lightness-contrast comparison,
%                  IDF3 ^= Lightness-structure comparison,
%                  IDF4 ^= Chroma comparison,
%                  IDF5 ^= Hue comparison,
%                  IDF6 ^= Chroma-contrast comparison
%            IDFConsts: Adjustment of each IDF to the working color space
%                       as well as weighting parameters
%               -> Array of length 6 with positive numbers for each IDF
%            IAM: Image appearance model for spatial filtering
%               -> Four IAMs implemented: 'iCAM_LAB2000HL', 'iCAM_YCC',
%                  'S-CIELAB' and 'NONE' like in the paper
%            CyclesPerDegree: Cycles per degree of visual field
%               -> Positive number
%            ScalesNum: Number of multi-scales used
%               -> Integer between 1 and 5
%            ScalesConsts: Weight of each scale
%               -> Positive number for each scale used
%            Combination: IDF combination model
%               -> One of three models (see paper for further information):
%                  'FAC' (factorial model)
%                  'LIN' (linear model)
%                  'HYB' (hybrid model)
%            CombOrder: Combination before or after averaging
%               -> Two orders possible: 'PIXEL' (before), 'MAP' (after)
%            CombConsts: Constants of IDF combination model
%               -> Positive number for each selected IDF
%            Window: 2-D filter for averaging within the sliding window
%               -> 2-D matrix with the same size as the sliding window
function [Prediction, Maps] = CID_advanced(Img1, Img2, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check input arguments
if (nargin < 2)
    error('IDM:vararginChk', ['Wrong number of input arguments! ' ...
        'Number must be at least ''2'' for both input images!']);
end
if (mod(nargin, 2))
    error('IDM:vararginChk', ['Wrong number of input arguments! ' ...
        'Number must be even']);
end

% Default parameters
Options = struct( ...
    'IDFChoice', [1, 2, 3, 4, 5], ...
    'IDFConsts', [0.002, 0.1, 0.1, 0.002, 0.008, 0.1], ...
    'IAM', 'S-CIELAB', ...
    'CyclesPerDegree', 20, ...
    'ScalesNum', 1, ...
    'ScalesConsts', [0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ...
    'Combination', 'FAC', ...
    'CombOrder', 'MAP', ...
    'CombConsts', [1, 1, 1, 1, 1, 1], ...
    'Window', fspecial('gaussian', 11, 1.5) ...
    );

% Set parameters from input
if (~isempty(varargin))
    for i = 1 : 2 : size(varargin, 2)
        if (strcmp(varargin{i}, 'IDFChoice'))
            Options.IDFChoice = unique(varargin{i+1});
        elseif (strcmp(varargin{i}, 'IDFConsts'))
            Options.IDFConsts = varargin{i+1};
        elseif (strcmp(varargin{i}, 'IAM'))
            Options.IAM = varargin{i+1};
        elseif (strcmp(varargin{i}, 'CyclesPerDegree'))
            Options.CyclesPerDegree = varargin{i+1};
        elseif (strcmp(varargin{i}, 'ScalesNum'))
            Options.ScalesNum = varargin{i+1};
        elseif (strcmp(varargin{i}, 'ScalesConsts'))
            Options.ScalesConsts = varargin{i+1};
        elseif (strcmp(varargin{i}, 'Combination'))
            Options.Combination = varargin{i+1};
        elseif (strcmp(varargin{i}, 'CombOrder'))
            Options.CombOrder = varargin{i+1};
        elseif (strcmp(varargin{i}, 'CombConsts'))
            Options.CombConsts = varargin{i+1};
        elseif (strcmp(varargin{i}, 'Window'))
            Options.Window = varargin{i+1};
        else
            warning('IDM:vararginChk', ['Wrong option ''%s''! Choose ' ...
                'between ''IDFChoice'', ''IDFConsts'', ''IAM'', ' ...
                '''CyclesPerDegree'', ''ScalesNum'', ''ScalesConsts'', '...
                '''Combination'', ''CombOrder'', ''CombConsts'' and ' ...
                '''Window'''], varargin{i});
        end
    end
end

% Check parameters
if (find(~ismember(Options.IDFChoice, 1:6)))
    error('IDM:parameterChk', ['Wrong choice of IDFs. Please enter ' ...
        'any combination of integers between 1 and 6!']);
end
if (length(Options.IDFConsts)~=6)
    error('IDM:parameterChk', ['Wrong array of IDF constants. Length of'...
        ' IDF constants array must be 6, even if not all IDFs are used.']);
end
if (min(Options.IDFConsts) <= 0)
    error('IDM:parameterChk', ['Wrong IDF constants. Please enter '...
        'a positive number for each IDF constant.']);
end
if (~ismember(Options.IAM, {'iCAM_LAB2000HL', 'iCAM_YCC', 'S-CIELAB', ...
        'NONE'}))
    error('IDM:parameterChk', ['Wrong image-appearance model. Choose '...
        'between ''iCAM_LAB2000HL'', ''iCAM_YCC'', ''S-CIELAB'' or ' ...
        '''NONE''.']);
end
if (Options.CyclesPerDegree <= 0)
    error('IDM:parameterChk', ['Wrong cycles per degree. Please enter '...
        'a positive number.']);
end
if (length(Options.ScalesNum) > 1)
    error('IDM:parameterChk', ['Wrong input for number of scales.' ...
        ' Please enter one integer between 1 and 5!']);
end
if (~ismember(Options.ScalesNum, 1:5))
    error('IDM:parameterChk', ['Wrong number of scales. Please ' ...
        'enter an integer between 1 and 5!']);
end
if (length(Options.ScalesConsts) < Options.ScalesNum)
    error('IDM:parameterChk', ['Wrong array of scale constants. Length' ...
        ' of scale-constants array must be the same as the number of ' ...
        'scales.']);
end
if (min(Options.ScalesConsts) < 0)
    error('IDM:parameterChk', ['Wrong scale constants. Please enter '...
        'a non-negative number for each scale constant.']);
end
if (~ismember(Options.Combination, {'FAC', 'LIN', 'HYB'}))
    error('IDM:parameterChk', ['Wrong combination model. Choose '...
        'between ''FAC'', ''LIN'' or ''HYB''.']);
end
if (~ismember(Options.CombOrder, {'MAP', 'PIXEL'}))
    error('IDM:parameterChk', ['Wrong combination model. Choose '...
        'between ''MAP'' or ''PIXEL''.']);
end
if (length(Options.CombConsts)~=6)
    error('IDM:parameterChk', ['Wrong array of combination constants. ' ...
        'Length of combination constants array must be 6, even if not ' ...
        'all IDFs are used.']);
end
if (min(Options.CombConsts) <= 0)
    error('IDM:parameterChk', ['Wrong combination constants. Please ' ...
        'enter a positive number for each combination constant.']);
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

% Transform image 1
Img1_XYZ = ImageSRGB2XYZ(Img1);
switch (Options.IAM)    % Apply image appearance model on XYZ data
    case 'iCAM_YCC'
        Img1_filt    = FilterImageCSF(Img1_XYZ, ...
            'cpd', Options.CyclesPerDegree, 'wrk_space', 'YCC-RIT', ...
            'pad_image', 'both');
    case 'iCAM_LAB2000HL'
        Img1_filt    = FilterImageCSF(Img1_XYZ, ...
            'cpd', Options.CyclesPerDegree, 'wrk_space', 'LAB00HL', ...
            'pad_image', 'both');
    case 'S-CIELAB'
        Img1_filt    = scielab_simple(2*Options.CyclesPerDegree, Img1_XYZ);
    case 'NONE'
        Img1_filt    = Img1_XYZ;
end
Img1_LAB2000HL = ImageXYZ2LAB2000HL(Img1_filt);

% Transform image 2
Img2_XYZ = ImageSRGB2XYZ(Img2);
switch (Options.IAM)    % Apply image appearance model on XYZ data
    case 'iCAM_YCC'
        Img2_filt    = FilterImageCSF(Img2_XYZ, ...
            'cpd', Options.CyclesPerDegree, 'wrk_space', 'YCC-RIT', ...
            'pad_image', 'both');
    case 'iCAM_LAB2000HL'
        Img2_filt    = FilterImageCSF(Img2_XYZ, ...
            'cpd', Options.CyclesPerDegree, 'wrk_space', 'LAB00HL', ...
            'pad_image', 'both');
    case 'S-CIELAB'
        Img2_filt    = scielab_simple(2*Options.CyclesPerDegree, Img2_XYZ);
    case 'NONE'
        Img2_filt    = Img2_XYZ;
end
Img2_LAB2000HL = ImageXYZ2LAB2000HL(Img2_filt);

%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE PREMAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculating the premaps is based upon the MSSIM from MeTriX MuX

% Prepare multi-scale approach
img1 = Img1_LAB2000HL;
img2 = Img2_LAB2000HL;
lod = [0.037828455507260; -0.023849465019560;  -0.110624404418440; ...
    0.377402855612830; 0.852698679008890;   0.377402855612830;  ...
    -0.110624404418440; -0.023849465019560; 0.037828455507260];
lpf = lod*lod';
lpf = lpf/sum(lpf(:)); % Low Pass filter from Biorthogonal 9/7 Wavelet
Premaps = cell(Options.ScalesNum, 8);

% Loop over all scales
for k = 1 : Options.ScalesNum

    % Downsampling with low pass filter
    if (k > 1)
        img1 = imfilter(img1, lpf, 'symmetric','same');
        img1 = img1(1:2:end,1:2:end,:);
        img2 = imfilter(img2, lpf, 'symmetric','same');
        img2 = img2(1:2:end,1:2:end,:);
    end
    
    % Abbreviations
    L1 = img1(:, :, 1);
    A1 = img1(:, :, 2);
    B1 = img1(:, :, 3);
    Chr1_sq = A1 .^2 +  B1 .^2;
    L2 = img2(:, :, 1);
    A2 = img2(:, :, 2);
    B2 = img2(:, :, 3);
    Chr2_sq = A2 .^2 + B2 .^2;
    
    % Mean intensity mu
    muL1 = filter2(Options.Window, L1, 'valid');
    muC1 = filter2(Options.Window, sqrt(Chr1_sq), 'valid');
    muL2 = filter2(Options.Window, L2, 'valid');
    muC2 = filter2(Options.Window, sqrt(Chr2_sq), 'valid');
    
    % Standard deviation sigma
    sL1_sq = filter2(Options.Window, L1 .^2, 'valid') - muL1 .^2;
    sL1_sq(sL1_sq < 0) = 0;
    sC1_sq = filter2(Options.Window, Chr1_sq, 'valid') - muC1 .^2;
    sC1_sq(sC1_sq < 0) = 0;
    sL2_sq = filter2(Options.Window, L2 .^2, 'valid') - muL2 .^2;
    sL2_sq(sL2_sq < 0) = 0;
    sC2_sq = filter2(Options.Window, Chr2_sq, 'valid') - muC2 .^2;
    sC2_sq(sC2_sq < 0) = 0;
   
    % Get mixed terms (dL_sq, dC_sq, dH_sq, sL12)
    dL_sq = (muL1 - muL2) .^2;
    dC_sq = (muC1 - muC2) .^2; 
    dH_sq = filter2(Options.Window, sqrt((A1 - A2) .^2 + (B1 - B2) .^2 - ...
        (sqrt(Chr1_sq) - sqrt(Chr2_sq)) .^2), 'valid') .^2;
    sL12 = filter2(Options.Window, L1 .* L2, 'valid') - muL1 .* muL2;
    
    % Write results in cell array
    Premaps{k, 1} = dL_sq;
    Premaps{k, 2} = dC_sq;
    Premaps{k, 3} = dH_sq;
    Premaps{k, 4} = sL1_sq;
    Premaps{k, 5} = sL2_sq;
    Premaps{k, 6} = sL12;
    Premaps{k, 7} = sC1_sq;
    Premaps{k, 8} = sC2_sq;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE MAPS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Maps_inv = cell(Options.ScalesNum, 6);
Maps = cell(Options.ScalesNum, 6);

% Loop over all scales
for k = 1 : Options.ScalesNum
    
    % Abbreviations
    sL1 = sqrt(Premaps{k, 4});
    sL2 = sqrt(Premaps{k, 5});
    sC1 = sqrt(Premaps{k, 7});
    sC2 = sqrt(Premaps{k, 8});
    
    if (k == Options.ScalesNum) % IDF 1) contributes only at the last scale
    % IDF 1) Lightness comparison
        Maps_inv{k, 1} = 1 ./ (Options.IDFConsts(1) * Premaps{k, 1} + 1);
        Maps{k, 1} = real(1 - Maps_inv{k, 1});
    end   
    
    % 2) Lightness-contrast comparison
        Maps_inv{k, 2} = (Options.IDFConsts(2) + 2 * sL1 .* sL2) ./ ...
            (Options.IDFConsts(2) + Premaps{k, 4} + Premaps{k, 5});
        Maps{k, 2} = real(1 - Maps_inv{k, 2});
    % 3) Lightness-structure comparison
        Maps_inv{k, 3} = (Options.IDFConsts(3) + Premaps{k, 6}) ./ ...
            (Options.IDFConsts(3) + sL1 .* sL2);
        Maps{k, 3} = real(1 - Maps_inv{k, 3});
        
    if (k == 1) % IDF 4) and IDF 5) contribute only at the first scale
    % 4) Chroma comparison
        Maps_inv{k, 4} = 1 ./ (Options.IDFConsts(4) * Premaps{k, 2} + 1);
        Maps{k, 4} = real(1 - Maps_inv{k, 4});
    % 5) Hue comparison
        Maps_inv{k, 5} = 1 ./ (Options.IDFConsts(5) * Premaps{k, 3} + 1);
        Maps{k, 5} = real(1 - Maps_inv{k, 5});
    end
    
    % 6) Chroma-contrast comparison
        Maps_inv{k, 6} = (Options.IDFConsts(6) + 2 * sC1 .* sC2) ./ ...
            (Options.IDFConsts(6) + Premaps{k, 7} + Premaps{k, 8});
        Maps{k, 6} = real(1 - Maps_inv{k, 6});
end

%%%%%%%%%%%%%%%%%%%%%%%%%% CALCULATE PREDICTION %%%%%%%%%%%%%%%%%%%%%%%%%%%

% Abbreviation
alpha = Options.ScalesConsts;
ScNum = Options.ScalesNum;
Const = Options.CombConsts;

% Restrict the number of scales on a total of NumScales and normalize
if (ScNum < length(alpha))
    alpha((ScNum + 1) : length(alpha)) = [];
end
alpha = alpha / sum(alpha);

% Exclude IDFs not chosen in Options.IDFChoice
Const(setdiff(1:6,Options.IDFChoice)) = 0;

% Combination order PIXEL only possible for LIN and HYB in single-scale
if ((strcmp(Options.CombOrder, 'PIXEL')) && (ScNum > 1))
    warning('IDM:combinationChk', ['Multi-scale approach not possible ' ...
        'with combination order PIXEL. Therefore, Combination order ' ...
        'was set to MAP. Single-scale approach would also be possible.']);
    Options.CombOrder = 'MAP';
end
    
% Combining IDFs before averaging
if (strcmp(Options.CombOrder, 'PIXEL'))
    
    % Compute predictions    
    if (strcmp(Options.Combination, 'FAC'))
        for k = 1 : ScNum
            Prediction = mean2(1 - Maps_inv{k, 1} .^Const(1) .* ...
            Maps_inv{k, 2} .^Const(2) .* Maps_inv{k, 3} .^Const(3) .* ...
            Maps_inv{k, 4} .^Const(4) .* Maps_inv{k, 5} .^Const(5) .* ...
            Maps_inv{k, 6} .^Const(6));
        end
        
    elseif (strcmp(Options.Combination, 'LIN'))
        Prediction = mean2(Maps{1, 1} * Const(1) + ...
                Maps{1, 2} * Const(2) + Maps{1, 3} * Const(3) + ...
                Maps{1, 4} * Const(4) + Maps{1, 5} * Const(5) + ...
                Maps{1, 6} * Const(6));
        
    elseif (strcmp(Options.Combination, 'HYB'))
        Prediction = mean2(1 - Maps{1, 1} .^Const(1) .* ...
            Maps{1, 2} .^Const(2) .* Maps{1, 3} .^Const(3) + ...
            1 - Maps{1, 4} .^Const(4) .* Maps{1, 6} .^Const(6) + ...
            1 - Maps{1, 5} .^Const(5));
    end    

% Combining IDFs after averaging
elseif (strcmp(Options.CombOrder, 'MAP'))
    
    % Image difference features
    IDF1 = zeros(1, 1);
    IDF2 = zeros(1, ScNum);
    IDF3 = zeros(1, ScNum);
    IDF4 = zeros(1, 1);
    IDF5 = zeros(1, 1);
    IDF6 = zeros(1, ScNum);
    
    % Compute mean before combining
    IDF1(1) = mean2(Maps_inv{ScNum, 1});    % Last scale
    
    [M, N] = size(Maps_inv{1, 2});   % Scale1
    NumEl = M*N;
    IDF2(1) = sum(Maps_inv{1, 2}(:)) / NumEl; % Faster than mean2 if used
    IDF3(1) = sum(Maps_inv{1, 3}(:)) / NumEl; %     more than once
    IDF4(1) = sum(Maps_inv{1, 4}(:)) / NumEl; %
    IDF5(1) = sum(Maps_inv{1, 5}(:)) / NumEl; %
    IDF6(1) = sum(Maps_inv{1, 6}(:)) / NumEl; %
    
    if (ScNum > 1)
        [M, N] = size(Maps_inv{2, 2});   % Scale2
        NumEl = M*N;
        IDF2(2) = sum(Maps_inv{2, 2}(:)) / NumEl;
        IDF3(2) = sum(Maps_inv{2, 3}(:)) / NumEl;
        IDF6(2) = sum(Maps_inv{2, 6}(:)) / NumEl;
    end
    
    if (ScNum > 2)
        [M, N] = size(Maps_inv{3, 2});   % Scale3
        NumEl = M*N;
        IDF2(3) = sum(Maps_inv{3, 2}(:)) / NumEl;
        IDF3(3) = sum(Maps_inv{3, 3}(:)) / NumEl;
        IDF6(3) = sum(Maps_inv{3, 6}(:)) / NumEl;
    end
    
    if (ScNum > 3)
        [M, N] = size(Maps_inv{4, 2});   % Scale4
        NumEl = M*N;
        IDF2(4) = sum(Maps_inv{4, 2}(:)) / NumEl;
        IDF3(4) = sum(Maps_inv{4, 3}(:)) / NumEl;
        IDF6(4) = sum(Maps_inv{4, 6}(:)) / NumEl;
    end
    
    if (ScNum > 4)
        [M, N] = size(Maps_inv{5, 2});   % Scale5
        NumEl = M*N;
        IDF2(5) = sum(Maps_inv{5, 2}(:)) / NumEl;
        IDF3(5) = sum(Maps_inv{5, 3}(:)) / NumEl;
        IDF6(5) = sum(Maps_inv{5, 6}(:)) / NumEl;
    end
    
    % Compute prediction by combining the means
    if (strcmp(Options.Combination, 'FAC'))
        Prediction = 1 - (IDF1^alpha(ScNum)) ^Const(1) * ...
            prod(IDF2.^alpha) ^Const(2) * prod(IDF3.^alpha) ^Const(3) * ...
            IDF4 ^Const(4) * IDF5 ^Const(5) * prod(IDF6.^alpha) ^Const(6);
        
    elseif (strcmp(Options.Combination, 'LIN'))
        Prediction = (1-IDF1^alpha(ScNum)) * Const(1) + ...
            (1-prod(IDF2.^alpha)) * Const(2) + ...
            (1-prod(IDF3.^alpha)) * Const(3) + (1-IDF4) * Const(4) + ...
            (1-IDF5) * Const(5) + (1-prod(IDF6.^alpha)) * Const(6);
        
    elseif (strcmp(Options.Combination, 'HYB'))
        Prediction = 1 - (IDF1^alpha(ScNum)) ^Const(1) * ...
            prod(IDF2.^alpha) ^Const(2) * prod(IDF3.^alpha) ^Const(3) + ...
            1 - IDF4 ^Const(4) * prod(IDF6.^alpha) ^Const(6) + ...
            1 - IDF5 ^Const(5);        
    end
end

% Occasionally, the prediction has a very small imaginary part; we keep
% only the real part of the prediction
Prediction = real(Prediction);

end