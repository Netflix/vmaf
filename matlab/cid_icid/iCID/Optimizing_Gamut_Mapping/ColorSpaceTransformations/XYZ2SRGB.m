% Transforms a set of XYZ values to SRGB coordinates.
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
%           SRGB = XYZ2SRGB(XYZ)
%
% Parameters:
%           XYZ       Nx3 matrix of XYZ values
%           SRGB      Nx3 matrix of SRGB values
%
% Example:
%           XYZ2SRGB([0.5 0.5 0.5])
%
function SRGB = XYZ2SRGB(XYZ)

% Note: All three sRGB conversion methods are invertible
% ('colorspace' is invertible with extremely small error)
conversion_method = 0;  % 0 = 'colorspace' function (Pascal Getreuer)
                        % 1 = MATLAB built-in
                        % 2 = Optprop toolbox (Jerker Wagberg)
                        % 3 = Spectral Toolbox

% Check input dimensions
assert((size(XYZ, 2) == 3) && (size(XYZ, 3) == 1), 'Input must be Kx3.');

switch (conversion_method)
    case 0  % 'colorspace' function (Pascal Getreuer)
        SRGB = colorspace('XYZ->RGB', XYZ);  
    case 1  % MATLAB built-in
        C = makecform('xyz2srgb', 'AdaptedWhitePoint', whitepoint('D65'));
        SRGB = applycform(XYZ, C);
    case 2  % Optprop toolbox (Jerker Wagberg)
        SRGB = xyz2rgb(XYZ .* 100, 'D65/2', 'srgb');
    case 3  % Spectral Toolbox
        M = [ 3.2406, -1.5372, -0.4986; ...
             -0.9689,  1.8758,  0.0415; ...
              0.0557, -0.2040,  1.0570];
        PROD = ((M * XYZ')');
        % Clip values to [0,1]
        SRGB = min(max(PROD, 0), 1);
        SRGB = (SRGB .* 12.92) .* (SRGB <= 0.0031308) ...
             + ((1.055 .* SRGB) .^ (1/2.4) - 0.055) .* (SRGB > 0.0031308);
         % Clip values to [0,1]
        SRGB = min(max(SRGB, 0), 1);
end