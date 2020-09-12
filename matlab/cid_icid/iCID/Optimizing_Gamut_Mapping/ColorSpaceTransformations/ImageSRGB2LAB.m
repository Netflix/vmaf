% Converts an image from sRGB to LAB coordinates.
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
%           IM_LAB = ImageSRGB2LAB(IM_SRGB)
%
% Parameters:
%           IM_SRGB         MxNx3 matrix of sRGB values
%           IM_LAB          MxNx3 matrix of LAB values
%
function IM_LAB = ImageSRGB2LAB(IM_SRGB)

    % Reshape the image before conversion
    [M, N, D] = size(IM_SRGB);
    IM_SRGB = reshape(IM_SRGB, [M * N, D]);
    % Perform the sRGB -> XYZ conversion
    IM_XYZ = SRGB2XYZ(IM_SRGB);
    % Perform the XYZ -> LAB conversion
    IM_LAB = XYZ2LAB(IM_XYZ);
    % Reshape the image after conversion
    IM_LAB = reshape(IM_LAB, [M, N, D]);