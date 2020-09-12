% Converts a set of LAB2000HL values to XYZ coordinates.
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
%           XYZ = LAB2000HL2XYZ(LAB2000HL)
%
% Parameters:
%           LAB2000HL       Nx3 matrix of LAB2000HL values
%           XYZ             Nx3 matrix of XYZ values
%
% Example:
%           LAB2000HL2XYZ([50 0 0])
function XYZ = LAB2000HL2XYZ(LAB2000HL)

% LAB2000HL -> LAB
LAB = LAB2000HL2LAB(LAB2000HL);

% LAB -> XYZ
XYZ = LAB2XYZ(LAB);