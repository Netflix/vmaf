% Transforms a set of LAB2000HL values to CIELAB.
%
% The transformation is based on lookup table interpolation. It is split
% into a one-dimensional lightness lookup table and a two-dimensional
% lookup table for the chromatic components.
%
% This is the inverse transformation. The forward transformation from
% CIELAB to LAB2000HL is implemented in 'LAB2LAB2000HL.m'.
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
%           LAB = LAB2000HL2LAB(LAB2000HL)
%
% Parameters:
%           LAB2000HL       Nx3 matrix of LAB2000HL values
%           LAB             Nx3 matrix of CIELAB values (WP: D65/10°)
%
% Example:
%           LAB = LAB2000HL2LAB([50 0 0])
function LAB = LAB2000HL2LAB(LAB2000HL)

%#ok<*NODEF>

% Check input dimensions
assert((size(LAB2000HL,2) == 3) && (size(LAB2000HL,3) == 1), ...
       'Input must be Nx3.');

% If necessary, clip input values
maxL = 75.153;                              % Maximum allowed lightness
LAB2000HL(LAB2000HL(:,1) <    0,1) =    0;  % Clip Lightness L*
LAB2000HL(LAB2000HL(:,1) > maxL,1) = maxL;  % Clip Lightness L*

load('LAB2000HL.mat', 'RegularGridInitInv', 'RegularGridInv', 'L');

LAB = LAB2000HL;
LAB(:,1) = interp1(L, 0:0.001:100, LAB2000HL(:,1));
LAB(:,2) = interp2(RegularGridInitInv(:,:,1), RegularGridInitInv(:,:,2), ...
                   RegularGridInv(:,:,1), LAB2000HL(:,2), LAB2000HL(:,3));
LAB(:,3) = interp2(RegularGridInitInv(:,:,1), RegularGridInitInv(:,:,2), ...
                   RegularGridInv(:,:,2), LAB2000HL(:,2), LAB2000HL(:,3));

% NaNs in the results?
assert(isempty(LAB(isnan(LAB))), ...
       'Resulting LAB values contain NaNs!');