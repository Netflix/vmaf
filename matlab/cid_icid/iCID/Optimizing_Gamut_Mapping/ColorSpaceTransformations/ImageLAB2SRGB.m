% Converts an image from the LAB color space to SRGB coordinates.
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
%           IM_SRGB = ImageLAB2SRGB(IM_LAB2000HL)
%
% Parameters:
%           IM_LAB          MxNx3 matrix of LAB values
%           IM_SRGB         MxNx3 matrix of SRGB values
%
function IM_SRGB = ImageLAB2SRGB(IM_LAB)

    % Reshape the image before conversion
    [M, N, D] = size(IM_LAB);
    IM_LAB = reshape(IM_LAB, [M * N, D]);
    % Perform the LAB -> XYZ transformation
    IM_XYZ = LAB2XYZ(IM_LAB);
    % Perform the XYZ -> SRGB transformation
    IM_SRGB = XYZ2SRGB(IM_XYZ);
    % Reshape the image after the transformation
    IM_SRGB = reshape(IM_SRGB, [M, N, D]);
    
end