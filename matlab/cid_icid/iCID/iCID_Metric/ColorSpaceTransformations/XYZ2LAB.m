% Transforms a set of XYZ values to CIELAB (D65/2° white point).
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
%           LAB = XYZ2LAB(XYZ)
%
% Parameters:
%           XYZ       Nx3 matrix of XYZ values
%           LAB       Nx3 matrix of CIELAB values
%
% Example:
%           XYZ2LAB([0.5 0.5 0.5])
%
function LAB = XYZ2LAB(XYZ)

% Check input dimensions
assert((size(XYZ, 2) == 3) && (size(XYZ, 3) == 1), 'Input must be Nx3.');

% XYZ -> CIELAB (WP: D65/2°)
LAB = colorspace('XYZ->Lab', XYZ);