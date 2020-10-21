% Converts an image from XYZ coordinates to the LAB2000HL color space.
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
% Authors:  Jens Preiss, Felipe Fernandes, Philipp Urban
%           Institute of Printing Science and Technology
%           Technische Universität Darmstadt
%           {preiss,fernandes,urban}@idd.tu-darmstadt.de
%           http://www.idd.tu-darmstadt.de/color
%
% Interface:
%           IM_LAB2000HL = ImageXYZ2LAB2000HL(IM_XYZ)
%
% Parameters:
%           IM_XYZ          MxNx3 matrix of XYZ values
%           IM_LAB2000HL    MxNx3 matrix of LAB2000HL values
%
function IM_LAB2000HL = ImageXYZ2LAB2000HL(IM_XYZ)
    % Reshape the image before conversion
    [M, N, D] = size(IM_XYZ);
    IM_XYZ = reshape(IM_XYZ, [M * N, D]);
    % Perform the XYZ -> LAB transformation
    IM_LAB = XYZ2LAB(IM_XYZ);
    % Perform the XYZ -> LAB2000HL transformation
    IM_LAB2000HL = LAB2LAB2000HL(IM_LAB);
    % Reshape the image after the transformation
    IM_LAB2000HL = reshape(IM_LAB2000HL, [M, N, D]);