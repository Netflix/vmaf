% Example for the use of the implementation of the improved 
% color-image-difference metric (iCID metric) which predicts the 
% perceived difference of two color images.
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

fprintf('\n%% \n%% Welcome to the example ''How to use iCID.m''!\n%% \n');

fprintf('\n%% Set MATLAB path if not yet done.\n\n');
disp('>> addpath(''ColorSpaceTransformations'');');
%%
addpath('ColorSpaceTransformations');
%%

fprintf(['\n%% Choose two sRGB images, Img1 and Img2.\n%% For each ' ...
    'image, pass a string containing the filename.\n\n']);
disp(['>> Img1 = ''ExampleImages/Image1.tif'';                      ' ...
    '% STRING']);
disp(['>> Img2 = ''ExampleImages/Image2.tif'';                      ' ...
    '% STRING']);
%%
Img1 = 'ExampleImages/Image1.tif';
Img2 = 'ExampleImages/Image2.tif';
%%

fprintf(['\n%% Alternatively, Img1 and Img2 can be image arrays of ' ...
         'size MxNx3.\n%% The data type can be UINT8 in the range of ' ...
         '[0,255] or DOUBLE in the range of [0,1].\n\n']);
disp(['>> Img1 = imread(''ExampleImages/Image1.tif'');              ' ...
    '% UINT8 in the range of [0,255]']);
disp(['>> Img2 = imread(''ExampleImages/Image2.tif'');              ' ...
    '% UINT8 in the range of [0,255]']);
disp(' ');
disp(['>> Img1 = double(imread(''ExampleImages/Image1.tif''))/255;  ' ...
    '% DOUBLE in the range of [0,1]']);
disp(['>> Img2 = double(imread(''ExampleImages/Image2.tif''))/255;  ' ...
    '% DOUBLE in the range of [0,1]']);

fprintf(['\n%% To compute predictions of the iCID measure with default' ...
    ' parameters, \n%% use the iCID function without additional ' ...
    'arguments.\n\n']);
disp('>> Prediction = iCID(Img1, Img2);');
disp('>> fprintf(''\nPrediction = %f\n'', Prediction);');
%%
[Prediction, Maps] = iCID(Img1, Img2);
fprintf('\nPrediction = %f\n', Prediction);
%%

fprintf(['\n%% The image-difference maps of the iCID measure are also ' ...
    'computed and put into a cell array (Maps).\n%% Maps{1} ' ...
    '= Lightness difference \n%% Maps{2} = Lightness contrast\n%% ' ...
    'Maps{3} = Lightness structure\n%% Maps{4} = Chroma difference ' ...
    '\n%% Maps{5} = Hue difference\n%% Maps{6} = Chroma contrast\n%% ' ...
    'Maps{7} = Chroma structure\n\n']);
disp('>> [Prediction, Maps] = iCID(Img1, Img2);');

fprintf(['\n%% Show the input images and each image-difference map ' ...
         'in a separate window.\n\n']);
disp('>> figure(1); imshow(Img1); title(''Image1.tif'');');
disp('>> figure(2); imshow(Img2); title(''Image2.tif'');');
disp('>> figure(3); imshow(Maps{1}); title(''Lightness difference'');');
disp('>> figure(4); imshow(Maps{2}); title(''Lightness contrast'');');
disp('>> figure(5); imshow(Maps{3}); title(''Lightness structure'');');
disp('>> figure(6); imshow(Maps{4}); title(''Chroma difference'');');
disp('>> figure(7); imshow(Maps{5}); title(''Hue difference'');');
disp('>> figure(8); imshow(Maps{6}); title(''Chroma contrast'');');
disp('>> figure(9); imshow(Maps{7}); title(''Chroma structure'');');
%%
figure(1); imshow(Img1); title('Image1.tif');
figure(2); imshow(Img2); title('Image2.tif');
figure(3); imshow(Maps{1}); title('Lightness difference');
figure(4); imshow(Maps{2}); title('Lightness contrast');
figure(5); imshow(Maps{3}); title('Lightness structure');
figure(6); imshow(Maps{4}); title('Chroma difference');
figure(7); imshow(Maps{5}); title('Hue difference');
figure(8); imshow(Maps{6}); title('Chroma contrast');
figure(9); imshow(Maps{7}); title('Chroma structure');
%%

fprintf('\n%% Save the maps as images.\n\n');
disp(['>> imwrite(Maps{1}, ''ExampleImages/Lightness_difference.tif'',' ...
    '''TIFF'');']);
disp(['>> imwrite(Maps{2}, ''ExampleImages/Lightness_contrast.tif'',' ...
    '''TIFF'');']);
disp(['>> imwrite(Maps{3}, ''ExampleImages/Lightness_structure.tif'',' ...
    '''TIFF'');']);
disp(['>> imwrite(Maps{4}, ''ExampleImages/Chroma_difference.tif'',' ...
    '''TIFF'');']);
disp(['>> imwrite(Maps{5}, ''ExampleImages/Hue_difference.tif'',' ...
    '''TIFF'');']);
disp(['>> imwrite(Maps{6}, ''ExampleImages/Chroma_contrast.tif'',' ...
    '''TIFF'');']);
disp(['>> imwrite(Maps{7}, ''ExampleImages/Chroma_structure.tif'',' ...
    '''TIFF'');']);
%%
imwrite(Maps{1}, 'ExampleImages/Lightness_difference.tif', 'TIFF');
imwrite(Maps{2}, 'ExampleImages/Lightness_contrast.tif', 'TIFF');
imwrite(Maps{3}, 'ExampleImages/Lightness_structure.tif', 'TIFF');
imwrite(Maps{4}, 'ExampleImages/Chroma_difference.tif', 'TIFF');
imwrite(Maps{5}, 'ExampleImages/Hue_difference.tif', 'TIFF');
imwrite(Maps{6}, 'ExampleImages/Chroma_contrast.tif', 'TIFF');
imwrite(Maps{7}, 'ExampleImages/Chroma_structure.tif', 'TIFF');
%%

fprintf(['\n%% If you want to change the parameters, type in the ' ...
    'parameter name and then the parameter value \n%% as a function ' ...
    'argument behind the input images.\n%% Some examples are ' ...
    'given below.\n%% For further information see the corresponding ' ...
    'm-file.\n\n']);
disp(['>> Prediction = iCID(Img1, Img2, ''Intent'', ' ...
    '''hue-preserving'');    % Prediction intent']);
disp('>> fprintf(''\nPrediction = %f\n'', Prediction);');
%%
Prediction = iCID(Img1, Img2, 'Intent', 'hue-preserving');
fprintf('\nPrediction = %f\n\n', Prediction);
%%

disp(['>> Prediction = iCID(Img1, Img2, ''IAM'', ''NONE''); ' ...
    '                % Image appearance model']);
disp('>> fprintf(''\nPrediction = %f\n'', Prediction);');
%%
Prediction = iCID(Img1, Img2, 'IAM', 'NONE');
fprintf('\nPrediction = %f\n\n', Prediction);
%%

disp(['>> Prediction = iCID(Img1, Img2, ''Omit_Maps67'', true); ' ...
    '         % Omit chroma contrast and chroma structure']);
disp('>> fprintf(''\nPrediction = %f\n'', Prediction);');
%%
Prediction = iCID(Img1, Img2, 'Omit_Maps67', true);
fprintf('\nPrediction = %f\n\n', Prediction);
%%

disp(['>> Prediction = iCID(Img1, Img2, ''Downsampling'', false); ' ...
    '        % Automatic Downsampling']);
disp('>> fprintf(''\nPrediction = %f\n'', Prediction);');
%%
Prediction = iCID(Img1, Img2, 'Downsampling', false);
fprintf('\nPrediction = %f\n\n', Prediction);
%%

disp(['>> Prediction = iCID(Img1, Img2, ''Intent'', ''chromatic'', ' ...
    '''IAM'', ''NONE'', ''Downsampling'', false);    % ' ...
    'Several parameters']);
disp('>> fprintf(''\nPrediction = %f\n'', Prediction);');
%%
Prediction = iCID(Img1, Img2, 'Intent', 'chromatic', 'IAM', 'NONE', ...
    'Downsampling', false);
fprintf('\nPrediction = %f\n\n', Prediction);
%%

fprintf('\n%% End of function example!\n');

end