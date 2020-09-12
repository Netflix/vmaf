% Example of use for 'CID.m' and 'CID_advanced.m'.
%
% This is an example for the use of the implementation of the algorithm for
% calculating the perceived image difference between two images - a color
% image difference (CID) measure.
%
% To start the example, type 'Example()' + ENTER in the Command Window or
% press F5 in the Editor.
%
% This code is supplementary material to the article:
%       I. Lissner, J. Preiss, P. Urban, M. Scheller Lichtenauer, and 
%       P. Zolliker, "Image-Difference Prediction: From Grayscale to
%       Color", IEEE Transactions on Image Processing (accepted), 2012
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
function Example()

fprintf(['\n%% \n%% Welcome to the example ''How to use CID.m and ' ...
         'CID_advanced.m''!\n%% \n']);

fprintf('\n%% Adjust MATLAB path if not yet done.\n\n');
disp('>> addpath(''MatlabFunctions/ColorSpaceTransformations'');');
disp('>> addpath(''MatlabFunctions/IAMFilters'');');
disp('>> addpath(''MatlabFunctions/IAMFilters/S-CIELAB'');');
addpath('MatlabFunctions/ColorSpaceTransformations', ...
        'MatlabFunctions/IAMFilters', ...
        'MatlabFunctions/IAMFilters/S-CIELAB');

fprintf(['\n%% Choose two sRGB images, Img1 and Img2.\n%% For each image, ' ...
         'pass a string containing the filename.\n\n']);
disp('>> Img1 = ''ExampleImages/Image1.tif'';                      % STRING');
disp('>> Img2 = ''ExampleImages/Image2.tif'';                      % STRING');
Img1 = 'ExampleImages/Image1.tif';
Img2 = 'ExampleImages/Image2.tif';

fprintf(['\n%% Alternatively, Img1 and Img2 can be image arrays of ' ...
         'size MxNx3.\n%% The data type can be UINT8 in the range of [0,255] or ' ...
         'DOUBLE in the range of [0,1].\n\n']);
disp(['>> Img1 = imread(''ExampleImages/Image1.tif'');              ' ...
      '% UINT8 in the range of [0,255]']);
disp(['>> Img2 = imread(''ExampleImages/Image2.tif'');              ' ...
      '% UINT8 in the range of [0,255]']);
disp(' ');
disp('>> Img1 = double(imread(''ExampleImages/Image1.tif''))/255;  % DOUBLE in the range of [0,1]');
disp('>> Img2 = double(imread(''ExampleImages/Image2.tif''))/255;  % DOUBLE in the range of [0,1]');

fprintf(['\n%% To compute predictions of the CID measure '...
    'with default parameters, use CID.m.\n\n']);
disp('>> Prediction = CID(Img1, Img2);');
disp('>> fprintf(''\nPrediction = %f\n'', Prediction);');
[Prediction, Maps] = CID(Img1, Img2);
fprintf('\nPrediction = %f\n', Prediction);

fprintf(['\n%% The maps of the image-difference features (IDFs) are also ' ...
    'computed and put\n%% into a cell array (Maps).\n%% Maps{1} = IDF1 ' ...
    '^= Lightness\n%% Maps{2} = IDF2 ^= Lightness-contrast\n%% Maps{3} = ' ...
    'IDF3 ^= Lightness-structure\n%% Maps{4} = IDF4 ^= Chroma\n%% Maps{5} = ' ...
    'IDF5 ^= Hue\n\n']);
disp('>> [Prediction, Maps] = CID(Img1, Img2);');

fprintf(['\n%% Show the input images and each image-difference map ' ...
         'in a separate window.\n\n']);
disp('>> figure(1); imshow(Img1); title(''Image1.tif'');');
disp('>> figure(2); imshow(Img2); title(''Image2.tif'');');
disp('>> figure(3); imshow(Maps{1}); title(''Lightness comparison'');');
disp('>> figure(4); imshow(Maps{2}); title(''Lightness-contrast comparison'');');
disp('>> figure(5); imshow(Maps{3}); title(''Lightness-structure comparison'');');
disp('>> figure(6); imshow(Maps{4}); title(''Chroma comparison'');');
disp('>> figure(7); imshow(Maps{5}); title(''Hue comparison'');');
figure(1); imshow(Img1); title('Image1.tif');
figure(2); imshow(Img2); title('Image2.tif');
figure(3); imshow(Maps{1}); title('Lightness comparison');
figure(4); imshow(Maps{2}); title('Lightness-contrast comparison');
figure(5); imshow(Maps{3}); title('Lightness-structure comparison');
figure(6); imshow(Maps{4}); title('Chroma comparison');
figure(7); imshow(Maps{5}); title('Hue comparison');

fprintf('\n%% Save the maps as images.\n\n');
disp('>> imwrite(Maps{1}, ''ExampleImages/Lightness_comparison.tif'', ''TIFF'');');
disp('>> imwrite(Maps{2}, ''ExampleImages/Lightness-contrast_comparison.tif'', ''TIFF'');');
disp('>> imwrite(Maps{3}, ''ExampleImages/Lightness-structure_comparison.tif'', ''TIFF'');');
disp('>> imwrite(Maps{4}, ''ExampleImages/Chroma_comparison.tif'', ''TIFF'');');
disp('>> imwrite(Maps{5}, ''ExampleImages/Hue_comparison.tif'', ''TIFF'');');
imwrite(Maps{1}, 'ExampleImages/Lightness_comparison.tif', 'TIFF');
imwrite(Maps{2}, 'ExampleImages/Lightness-contrast_comparison.tif', 'TIFF');
imwrite(Maps{3}, 'ExampleImages/Lightness-structure_comparison.tif', 'TIFF');
imwrite(Maps{4}, 'ExampleImages/Chroma_comparison.tif', 'TIFF');
imwrite(Maps{5}, 'ExampleImages/Hue_comparison.tif', 'TIFF');

fprintf(['\n%% The function CID.m uses default parameters. If you ' ...
         'want to change the parameters,\n%% use CID_advanced.m ' ...
         'instead; some examples are given below. For further ' ...
         'information\n%% see the corresponding m-file.\n\n']);
disp(['>> Prediction = CID_advanced(Img1, Img2, ''IAM'', ''iCAM_YCC''); ' ...
    '    % Image appearance model']);
Prediction = CID_advanced(Img1, Img2, 'IAM', 'iCAM_YCC');
fprintf('Prediction = %f\n', Prediction);
disp(['>> Prediction = CID_advanced(Img1, Img2, ''CyclesPerDegree'', 40); ' ...
      '% Degree of visual field']);
Prediction = CID_advanced(Img1, Img2, 'CyclesPerDegree', 40);
fprintf('Prediction = %f\n', Prediction);
disp(['>> Prediction = CID_advanced(Img1, Img2, ''ScalesNum'', 5); ' ...
      '       % Multi-scale with 5 scales']);
Prediction = CID_advanced(Img1, Img2, 'ScalesNum', 5);
fprintf('Prediction = %f\n', Prediction);
disp(['>> Prediction = CID_advanced(Img1, Img2, ''ScalesNum'', 5, ''IAM'', ' ...
      '''NONE''); % More than one parameter changed']);
Prediction = CID_advanced(Img1, Img2, 'ScalesNum', 5, 'IAM', 'NONE');
fprintf('Prediction = %f\n', Prediction);

fprintf('\n%% End of example!\n');

end