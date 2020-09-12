% Transforms a set of CIELAB values to XYZ coordinates (D65/2° white p.).
%
% This function uses the 'colorspace' package by Pascal Getreuer.
% Please see the README.txt file for further information.
%
% This code is supplementary material to the article:
%           I. Lissner, J. Preiss, P. Urban, M. Scheller Lichtenauer, and 
%           P. Zolliker, "Image-Difference Prediction: From Grayscale to
%           Color", IEEE Transactions on Image Processing (accepted), 2012.
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
%           XYZ = LAB2XYZ(LAB)
%
% Parameters:
%           LAB       Nx3 matrix of CIELAB values
%           XYZ       Nx3 matrix of XYZ values
%
% Example:
%           LAB2XYZ([100 0 0])
function XYZ = LAB2XYZ(LAB)

% Check input dimensions
assert((size(LAB,2) == 3) && (size(LAB,3) == 1), 'Input must be Nx3.');

% Check lightness values
assert(isempty(LAB(LAB(:,1) > 100)), 'Lightness L* must be in [0,100].');

% CIELAB -> XYZ (WP: D65/2°)
XYZ = colorspace('Lab->XYZ', LAB);