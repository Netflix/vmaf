% Transforms a set of CIELAB values to XYZ coordinates (D65/2° white p.).
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
%           XYZ = LAB2XYZ(LAB)
%
% Parameters:
%           LAB       Nx3 matrix of CIELAB values
%           XYZ       Nx3 matrix of XYZ values
%
% Example:
%           LAB2XYZ([100 0 0])
%
function XYZ = LAB2XYZ(LAB)

% Check input dimensions
assert((size(LAB, 2) == 3) && (size(LAB, 3) == 1), 'Input must be Nx3.');

% Check lightness values
assert(isempty(LAB(LAB(:, 1) > 100)), 'Lightness L* must be in [0, 100].');

% CIELAB -> XYZ (WP: D65/2°)
XYZ = colorspace('Lab->XYZ', LAB);