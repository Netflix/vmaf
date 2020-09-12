% Implementation of the optimizing gamut-mapping (OGM) algorithm which
% optimizes a gamut-mapped image by minimizing the iCID measure of the
% original image and the gamut-mapped image
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
% Input:  (1) OriginalImg: Original sRGB image to be gamut mapped (string
%                          with path and filename)
%         (2) ICCProfile: ICC profile of the gamut in which the original
%                         image is mapped (string with path and filename).
%                         A mat-file of the gamut is created if not yet
%                         done which then can be entered directly. In this
%                         case a mapped image as input is obligatory.
%  (optional) Intent: Optimizing intent (choose between 'perceptual'
%                     [default], 'hue-preserving', and 'chromatic')
%  (optional) MappedImg: An already gamut-mapped image to be optimized.
%                        If no gamut-mapped image is entered a perceptual
%                        ICC-profile gamut mapping is used instead (string 
%                        with path and filename)
%
% Output: (1) Optimized_SRGB: Optimized gamut-mapped sRGB image
%         (2) Mapped_SRGB: Gamut-mapped SRGB image on which the 
%                          optimization was applied
%
% Example: OptimizedImg = Optimizing_Gamut_Mapping(OriginalImg, ...
%                             ICCProfile, 'Intent', 'chromatic');
%
function [Optimized_SRGB, Mapped_SRGB] = Optimizing_Gamut_Mapping( ...
    OriginalImg, ICCProfile, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% SETTINGS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Check input arguments
if (nargin < 2)
    error('OGM:vararginChk', ['Wrong number of input arguments! ' ...
        'Number must be at least ''2'' for the original image and ' ...
        'the gamut!']);
end
if (mod(nargin, 2))
    error('iCID:vararginChk', ['Wrong number of input arguments! ' ...
        'Number must be even!']);
end

% Default parameters
Intent = 'perceptual';
MappedImg = '';

% Set parameters from input
if (~isempty(varargin))
    for i = 1 : 2 : size(varargin, 2)
        if (strcmp(varargin{i}, 'Intent'))
            Intent = varargin{i+1};
        elseif (strcmp(varargin{i}, 'MappedImg'))
            MappedImg = varargin{i+1};
        else
            warning('IDM:vararginChk', ['Wrong option ''%s''! Choose ' ...
                'between ''Intent'' and ''MappedImg''. Default value ' ...
                'is set.'], varargin{i});
        end
    end
end

% Intent mexw64 number
if (strcmp(Intent, 'hue-preserving'))
    IntentNum = 0;
elseif (strcmp(Intent, 'perceptual'))
    IntentNum = 1;
elseif (strcmp(Intent, 'chromatic'))
    IntentNum = 2;
else
    warning('OGM:IntentChk', ['Wrong optimization intent entered. '...
        'Please choose between ''hue-preserving'', ''perceptual'', ' ...
        'and ''chromatic''. The optimization intent is set to the ' ...
        'default value (''hue-preserving'').']);
    IntentNum = 0;
end

% Set constants
LScale    = 1.6899;
ABScale   = 2.6771;
ABOffset  = 128;

%%%%%%%%%%%%%%%%% PREPARE ORIGINAL AND GAMUT-MAPPED IMAGE %%%%%%%%%%%%%%%%%

% Load image and transform as mexw64 output
Original_SRGB = imread(OriginalImg);
if (size(Original_SRGB, 3) == 4) % For sRGB TIFF-files in MATLAB the fourth
    Original_SRGB(:, :, 4) = [];   % dimension is omitted
end
Original_LAB       = ImageSRGB2LAB(Original_SRGB);
Original_LAB2000HL = ImageLAB2LAB2000HL(Original_LAB);
Original_UINT8     = DOUBLE2UINT8(Original_LAB2000HL, ...
    LScale, ABScale, ABOffset);

% Get gamut-mapped image and transform as mexw64 output
if (MappedImg)
    Mapped_SRGB = imread(MappedImg);
    if (size(Mapped_SRGB, 3) == 4) % For sRGB TIFF-files in MATLAB the
        Mapped_SRGB(:, :, 4) = [];   % fourth dimension is omitted
    end
    if (sum(size(Mapped_SRGB) ~= size(Original_SRGB)))
        error('OGM:ImgSizeChk', ['Wrong size of gamut-mapped image. ' ...
            'The image size of the original and the gamut-mapped image' ...
            ' must be equal.']);
    end
    Mapped_LAB = ImageSRGB2LAB(Mapped_SRGB);
else
    [pathstricc, nameicc, exticc] = fileparts(ICCProfile);
    if (strcmp(exticc, '.mat'))
        error('OGM:ICCProfileChk', ['The icc profile is a mat-file. ' ...
            'This is only possible if a gamut-mapped image is entered.' ...
            ' Please use do not use a mat-file in this case or enter ' ...
            'a mat-file.']);
    end
    [M, N, D] = size(Original_LAB);
    Mapped_LAB = reshape(Original_LAB, M * N, D);
    C = makecform('clut', iccread(ICCProfile), 'BToA0');
    Mapped_PCS = applycform(Mapped_LAB, C)';
    C = makecform('clut', iccread(ICCProfile), 'AToB1');
    Mapped_LAB = applycform(Mapped_PCS', C)';
    Mapped_LAB = reshape(Mapped_LAB', M, N, D);
    Mapped_SRGB = ImageLAB2SRGB(Mapped_LAB);
    [pathstr, name] = fileparts(OriginalImg);
    imwrite(Mapped_SRGB, [pathstr filesep name '_' nameicc '_Mapped.tif'], 'TIFF');   
end
Mapped_LAB2000HL = ImageLAB2LAB2000HL(Mapped_LAB);
Mapped_UINT8     = DOUBLE2UINT8(Mapped_LAB2000HL, ...
    LScale, ABScale, ABOffset);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAMUT MAPPING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load gamut information
Gamut = MakeGamut(ICCProfile);

% Interface with mexw64
fprintf(['\n%% RUNNING GAMUT-MAPPING OPTIMIZATION. THIS MAY TAKE ' ...
    'SEVERAL MINUTES\n']);
Optimized_UINT8 = OptimizingGamutMapping(Original_UINT8, ...
    Mapped_UINT8, Gamut, uint8(IntentNum));
clear mex;

%%%%%%%%%%%%%%%%%%%%% SRGB OUTPUT OF OPTIMIZED IMAGE %%%%%%%%%%%%%%%%%%%%%%

% Transform back to SRGB
Optimized_LAB2000HL = UINT82DOUBLE(Optimized_UINT8, ...
    LScale, ABScale, ABOffset);
    
Optimized_SRGB  = ImageLAB2000HL2SRGB(Optimized_LAB2000HL);

end

function Img_UINT8 = DOUBLE2UINT8(Img_DOUBLE, LScale, ABScale, ABOffset)

Img_UINT8           = zeros(size(Img_DOUBLE));
Img_UINT8(:, :, 1)  = Img_DOUBLE(:, :, 1) * LScale;
Img_UINT8(:, :, 2)  = Img_DOUBLE(:, :, 2) * ABScale + ABOffset;
Img_UINT8(:, :, 3)  = Img_DOUBLE(:, :, 3) * ABScale + ABOffset;
Img_UINT8           = uint8(Img_UINT8);

end

function Img_DOUBLE = UINT82DOUBLE(Img_UINT8, LScale, ABScale, ABOffset)

Img_DOUBLE          = zeros(size(Img_UINT8));
Img_DOUBLE(:, :, 1) = double(Img_UINT8(:, :, 1)) / LScale;
Img_DOUBLE(:, :, 2) = (double(Img_UINT8(:, :, 2)) - ABOffset) / ABScale;
Img_DOUBLE(:, :, 3) = (double(Img_UINT8(:, :, 3)) - ABOffset) / ABScale;

end

function Gamut = MakeGamut(ICCProfile)

[pathstr, name] = fileparts(ICCProfile);
MATFile = fullfile(pathstr, [name '.mat']);

if (~exist(MATFile, 'file'))
    fprintf(['\n%% CREATING MAT-FILE OF THE GAMUT FOR FASTER ' ...
        'COMPUTATION.\n%% SAVED IN SAME PATH AS THE ICC PROFILE.\n']);
    disp('');
    LScale = 1.6899;
    ABScale = 2.6771;
    Offset = 128;
    P = iccread(ICCProfile);
    C = makecform('clut', P, 'AToB1');
    [X1, X2, X3] = ndgrid(0:255, 0:255, 0:255);
    Colorants = [reshape(X1, 256^3, 1), reshape(X2, 256^3, 1), ...
        reshape(X3, 256^3, 1), zeros(256^3, 1)] / 255;
    Lab = single(applycform(Colorants, C))';
    clear X1 X2 X3;
    [X1, X2, X3, X4] = ndgrid(0:5:255, 0:5:255, 0:5:255, 0:5:255);
    Colorants = [reshape(X1, 52^4, 1), reshape(X2, 52^4, 1), ...
        reshape(X3, 52^4, 1), reshape(X4, 52^4, 1)] / 255;
    Lab = [Lab, single(applycform(Colorants, C))'];
    clear X1 X2 X3 X4 Colorants;
    Lab = LAB2LAB2000HL(Lab');
    Lab = bitor(bitor(bitshift(uint32(round(Lab(:, 1) * LScale)), 16), ...
        bitshift(uint32(round(Lab(:, 2) * ABScale + Offset)), 8)), ...
        uint32(round(Lab(:, 3) * ABScale + Offset)));
    GAMUT = zeros(256 * 256 * 128, 1, 'int8');
    GAMUT(Lab) = 1;    
    save(MATFile, 'GAMUT');
else
    load(MATFile, 'GAMUT');
end

Gamut = GAMUT;
end
