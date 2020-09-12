% Converts an image from sRGB to XYZ coordinates.
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
%           IM_XYZ = ImageSRGB2XYZ(IM_SRGB)
%
% Parameters:
%           IM_SRGB         MxNx3 matrix of sRGB values
%           IM_XYZ          MxNx3 matrix of XYZ values
function IM_XYZ = ImageSRGB2XYZ(IM_SRGB)
    % Reshape the image before conversion
    [M,N,D] = size(IM_SRGB);
    IM_SRGB = reshape(IM_SRGB, [M*N,D]);
    % Perform the sRGB -> XYZ conversion
    IM_XYZ = SRGB2XYZ(IM_SRGB);
    % Reshape the image after conversion
    IM_XYZ = reshape(IM_XYZ, [M,N,D]);
end