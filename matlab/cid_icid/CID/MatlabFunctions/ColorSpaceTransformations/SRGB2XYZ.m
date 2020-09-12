% Transforms a set of sRGB values to XYZ coordinates.
%
% This function uses the 'colorspace' package by Pascal Getreuer.
% Please see the README.txt file for further information.
%
% This code is supplementary material to the article:
%           Lissner, Preiss, Urban, Scheller Lichtenauer, Zolliker,
%           "Image-Difference Prediction: From Grayscale to Color",
%           IEEE Transactions on Image Processing (accepted), 2012.
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
% Interface:
%           XYZ = SRGB2XYZ(SRGB)
%
% Parameters:
%           SRGB      Nx3 matrix of sRGB values
%           XYZ       Nx3 matrix of XYZ values
%
% Example:
%           SRGB2XYZ([1 0 0])
function XYZ = SRGB2XYZ(SRGB)

% Check input dimensions
assert((size(SRGB,2) == 3) && (size(SRGB,3) == 1), 'Input must be Nx3.');

% sRGB -> XYZ (WP: D65/2°)
XYZ = colorspace('RGB->XYZ', SRGB);