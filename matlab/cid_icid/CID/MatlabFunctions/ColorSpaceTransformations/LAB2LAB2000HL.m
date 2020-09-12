% Transforms a set of CIELAB values to the highly perceptually uniform
% hue linear LAB2000HL color space (the CIEDE2000 color-difference formula
% serves as a reference of perceptual uniformity).
%
% The transformation is based on lookup table interpolation. It is split
% into a one-dimensional lightness lookup table and a two-dimensional
% lookup table for the chromatic components.
%
% This is the forward transformation. The inverse transformation from
% LAB2000HL to CIELAB is implemented in 'LAB2000HL2LAB.m'.
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
%           LAB2000HL = LAB2LAB2000HL(LAB)
%
% Parameters:
%           LAB             Nx3 matrix of CIELAB values (WP: D65/10°)
%           LAB2000HL       Nx3 matrix of LAB2000HL values
%
% Example:
%           LAB2000HL = LAB2LAB2000HL([100 0 0])
function LAB2000HL = LAB2LAB2000HL(LAB)

% Check input dimensions
assert((size(LAB,2) == 3) && (size(LAB,3) == 1), 'Input must be Nx3.');

% If necessary, clip input values
LAB(LAB(:,1) <    0,1) =    0;	% Clip Lightness L*
LAB(LAB(:,1) >  100,1) =  100;  % Clip Lightness L*
LAB(LAB(:,2) < -128,2) = -128;  % Clip R/G axis a*
LAB(LAB(:,2) >  128,2) =  128;  % Clip R/G axis a*
LAB(LAB(:,3) < -128,3) = -128;  % Clip B/Y axis b*
LAB(LAB(:,3) >  128,3) =  128;  % Clip B/Y axis b*

load('LAB2000HL.mat', 'RegularGridInit', 'RegularGrid', 'L');

LAB2000HL = LAB;
LAB2000HL(:,1) = interp1(0:0.001:100, L, LAB(:,1));
LAB2000HL(:,2) = interp2(RegularGridInit(:,:,1), RegularGridInit(:,:,2), ...
                         RegularGrid(:,:,1), LAB(:,2), LAB(:,3));
LAB2000HL(:,3) = interp2(RegularGridInit(:,:,1), RegularGridInit(:,:,2), ...
                         RegularGrid(:,:,2), LAB(:,2), LAB(:,3));
                     
% NaNs in the results?
assert(isempty(LAB2000HL(isnan(LAB2000HL))), ...
       'Resulting LAB2000HL values contain NaNs!');