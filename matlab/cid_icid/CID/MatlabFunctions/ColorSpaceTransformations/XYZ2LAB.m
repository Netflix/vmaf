% Transforms a set of XYZ values to CIELAB (D65/2° white point).
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
%           LAB = XYZ2LAB(XYZ)
%
% Parameters:
%           XYZ       Nx3 matrix of XYZ values
%           LAB       Nx3 matrix of CIELAB values
%
% Example:
%           XYZ2LAB([0.5 0.5 0.5])
function LAB = XYZ2LAB(XYZ)

% Check input dimensions
assert((size(XYZ,2) == 3) && (size(XYZ,3) == 1), 'Input must be Nx3.');

% XYZ -> CIELAB (WP: D65/2°)
LAB = colorspace('XYZ->Lab', XYZ);