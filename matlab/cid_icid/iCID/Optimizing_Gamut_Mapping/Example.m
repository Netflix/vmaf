% Example for the use of the implementation of the optimizing gamut-
% mapping algorithm which optimizes a gamut-mapped image by minimizing 
% the iCID measure of the original image and the gamut-mapped image
%
% 2014/02/27: Version 1.00
%
% To start the example, type 'Example()' + ENTER in the Command Window or
% press F5 in the Editor.
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
function Example()

fprintf(['\n%% \n%% Welcome to the example ''How to use ' ...
    'Optimizing_Gamut_Mapping.m''!\n%% \n']);

fprintf('\n%% Set MATLAB path if not yet done.\n\n');
disp('>> addpath(''ColorSpaceTransformations'');');
disp('>> addpath(''Optimization'');');
%%
addpath('ColorSpaceTransformations');
addpath('Optimization');
%%

fprintf(['\n%% We need the filename of both the original image and ' ...
    'the ICC profile which \n%% describes the output gamut.\n\n']);
disp(['>> OriginalImage = ''ExampleImages/Drops.tif'';          ' ...
    ' % STRING']);
disp(['>> ICCProfile    = ''ExampleICCProfiles/USNewsprintSNAP2007' ...
    '.icc''; % STRING']);
%%
OriginalImage = 'ExampleImages/Drops.tif';
ICCProfile    = 'ExampleICCProfiles/USNewsprintSNAP2007.icc';
%%

fprintf(['\n%% FIRST, we use ''Optimizing_Gamut_Mapping.m'' for ' ...
    'optimizing an already gamut-mapped image.\n%% Without an ' ...
    'additional argument the perceptual optimization intent is ' ...
    'used.\n\n']);
disp(['>> MappedImage     = ''ExampleImages/Drops_USNewsprintSNAP2007_' ...
    'Mapped.tif''; % STRING']);
disp(['>> Optimized_SRGB = Optimizing_Gamut_Mapping(OriginalImage, ' ...
    'ICCProfile, ''MappedImg'', MappedImage);']);
%%
MappedImage     = 'ExampleImages/Drops_USNewsprintSNAP2007_Mapped.tif';
Optimized_SRGB = Optimizing_Gamut_Mapping(OriginalImage, ICCProfile, ...
    'MappedImg', MappedImage);
%%

fprintf(['\n%% Show the original, the gamut-mapped, and the optimized ' ...
    'image in a single window.\n\n']);
disp('>> Original_SRGB = imread(OriginalImage);');
disp('>> Mapped_SRGB = imread(MappedImage);');
disp(['>> if (size(Original_SRGB, 3) == 4), Original_SRGB(:, :, 4) = ' ...
    '[]; end % For sRGB TIFF-files in MATLAB the fourth dimension is ' ...
    'omitted']);
disp('>> figure(''position'', [100 100 950 750]);');
disp('>> subplot(1,3,1); imshow(Original_SRGB); title(''Original'');');
disp('>> subplot(1,3,2); imshow(Mapped_SRGB); title(''ICC profile'');');
disp('>> imshow(Optimized_SRGB); title(''iCID optimized'');');
%%
Original_SRGB = imread(OriginalImage);
if (size(Original_SRGB, 3) == 4), Original_SRGB(:, :, 4) = []; end
Mapped_SRGB = imread(MappedImage);
if (size(Mapped_SRGB, 3) == 4), Mapped_SRGB(:, :, 4) = []; end
figure('position', [100 100 950 750]);
subplot(1,3,1); imshow(Original_SRGB); title('Original');
subplot(1,3,2); imshow(Mapped_SRGB); title('ICC profile');
subplot(1,3,3); imshow(Optimized_SRGB); title('iCID optimized');
%%

fprintf('\n%% Save the optimized image.\n\n');
disp(['>> imwrite(Optimized_SRGB, ''ExampleImages/' ...
    'Drops_USNewsprintSNAP2007_Optimized.tif'', ''TIFF'');']);
%%
imwrite(Optimized_SRGB, ['ExampleImages/' ...
    'Drops_USNewsprintSNAP2007_Optimized.tif'], 'TIFF');
%%


%%
fprintf(['\n\n%% !!!Please note that the following optimizations are ' ...
    'not executed to finish the example faster.\n\n']);
%%

fprintf(['\n%% If you want to change the optimization intent, type in ' ...
    '''Intent'' and then ''perceptual'' (default),\n%% ''hue-' ...
    'preserving'', or ''chromatic'' behind the original image and ' ...
    'the gamut.\n%% For further information see the corresponding ' ...
    'm-file.\n\n']);
disp(['>> Optimized_SRGB_perceptual    = Optimizing_Gamut_Mapping(' ...
    'OriginalImage, ICCProfile, ''Intent'', ''perceptual'', ' ...
    '''MappedImg'', MappedImage);']);
disp(['>> Optimized_SRGB_huepreserving = Optimizing_Gamut_Mapping(' ...
    'OriginalImage, ICCProfile, ''Intent'', ''hue-preserving'', ' ...
    '''MappedImg'', MappedImage);']);
disp(['>> Optimized_SRGB_chromatic    = Optimizing_Gamut_Mapping(' ...
    'OriginalImage, ICCProfile, ''Intent'', ''chromatic'', ' ...
    '''MappedImg'', MappedImage);']);
%%
%%

fprintf(['\n%% SECOND, we use ''Optimizing_Gamut_Mapping.m'' as a ' ...
    'complete gamut-mapping algorithm.\n%% For this purpose, we ' ...
    'only enter the original image and the icc profile.\n\n']);
disp(['>> Optimized2_SRGB = Optimizing_Gamut_Mapping(OriginalImage, ' ...
    'ICCProfile);']);
%%
%%

fprintf(['\n%% Again, we can set the optimization intent. For ' ...
    'instance:\n\n' '>> Optimized2_SRGB_chromatic = ' ...
    'Optimizing_Gamut_Mapping(OriginalImage, ICCProfile, ''Intent'', ' ...
    '''chromatic'');\n']);
%%
%%

fprintf(['\n%% Before applying the gamut-mapping optimization, a ' ...
    'gamut-mapped image is computed. \n%% This is done in the profile ' ...
    'connection space by an ICC profile with perceptual \n%% rendering' ...
    ' intent. The gamut-mapped image is the second output argument.\n\n']);
disp(['>> [Optimized2_SRGB, Mapped2_SRGB] = Optimizing_Gamut_Mapping' ...
    '(OriginalImage, ICCProfile, ''Intent'', ''chromatic'');']);
%%
%%

fprintf('\n%% End of function example!\n');

end