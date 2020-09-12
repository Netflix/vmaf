% Converts an image from the LAB2000HL color space to XYZ coordinates.
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
%           IM_XYZ = ImageLAB2000HL2XYZ(IM_LAB2000HL)
%
% Parameters:
%           IM_LAB2000HL    MxNx3 matrix of LAB2000HL values
%           IM_XYZ          MxNx3 matrix of XYZ values
function IM_XYZ = ImageLAB2000HL2XYZ(IM_LAB2000HL)
    % Reshape the image before conversion
    [M,N,D] = size(IM_LAB2000HL);
    IM_LAB2000HL = reshape(IM_LAB2000HL, [M*N,D]);
    % Perform the LAB2000HL -> XYZ transformation
    IM_XYZ = LAB2000HL2XYZ(IM_LAB2000HL);
    % Reshape the image after the transformation
    IM_XYZ = reshape(IM_XYZ, [M,N,D]);
end