% Transforms a set of sRGB values to XYZ coordinates.
%
% This function uses the 'colorspace' package by Pascal Getreuer.
% Please see the README.txt file for further information.
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
% Interface:
%           XYZ = SRGB2XYZ(SRGB)
%
% Parameters:
%           SRGB      Nx3 matrix of sRGB values
%           XYZ       Nx3 matrix of XYZ values
%
% Example:
%           SRGB2XYZ([1 0 0])
%
function XYZ = SRGB2XYZ(SRGB)

% Check input dimensions
assert((size(SRGB, 2) == 3) && (size(SRGB, 3) == 1), 'Input must be Nx3.');

% sRGB -> XYZ (WP: D65/2°)
XYZ = colorspace('RGB->XYZ', SRGB);