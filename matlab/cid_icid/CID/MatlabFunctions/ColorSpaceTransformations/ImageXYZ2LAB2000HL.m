% Converts an image from XYZ coordinates to the LAB2000HL color space.
%
% For more information on the LAB2000HL color space, please see:
%           I. Lissner and P. Urban, "Toward a Unified Color Space
%           for Perception-Based Image Processing", IEEE Transactions
%           on Image Processing, Vol. 21, No. 3, pp. 1153-1168, 2012.
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
%           IM_LAB2000HL = ImageXYZ2LAB2000HL(IM_XYZ)
%
% Parameters:
%           IM_XYZ          MxNx3 matrix of XYZ values
%           IM_LAB2000HL    MxNx3 matrix of LAB2000HL values
function IM_LAB2000HL = ImageXYZ2LAB2000HL(IM_XYZ)
    % Reshape the image before conversion
    [M,N,D] = size(IM_XYZ);
    IM_XYZ = reshape(IM_XYZ, [M*N,D]);
    % Perform the XYZ -> LAB2000HL transformation
    IM_LAB2000HL = XYZ2LAB2000HL(IM_XYZ);
    % Reshape the image after the transformation
    IM_LAB2000HL = reshape(IM_LAB2000HL, [M,N,D]);
end