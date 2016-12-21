% RES = pointOp(IM, LUT, ORIGIN, INCREMENT, WARNINGS)
%
% Apply a point operation, specified by lookup table LUT, to image IM.
% LUT must be a row or column vector, and is assumed to contain
% (equi-spaced) samples of the function.  ORIGIN specifies the
% abscissa associated with the first sample, and INCREMENT specifies the
% spacing between samples.  Between-sample values are estimated via
% linear interpolation.  If WARNINGS is non-zero, the function prints
% a warning whenever the lookup table is extrapolated.
%
% This function is much faster than MatLab's interp1, and allows
% extrapolation beyond the lookup table domain.  The drawbacks are
% that the lookup table must be equi-spaced, and the interpolation is
% linear.

% Eero Simoncelli, 8/96.

function res = pointOp(im, lut, origin, increment, warnings)

%% NOTE: THIS CODE IS NOT ACTUALLY USED! (MEX FILE IS CALLED INSTEAD)

fprintf(1,'WARNING: You should compile the MEX version of "pointOp.c",\n         found in the MEX subdirectory of matlabPyrTools, and put it in your matlab path.  It is MUCH faster.\n');

X = origin + increment*[0:size(lut(:),1)-1];
Y = lut(:);

res = reshape(interp1(X, Y, im(:), 'linear', 'extrap'),size(im));

