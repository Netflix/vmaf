% RES = spyrHigh(PYR, INDICES)
%
% Access the highpass residual band from a steerable pyramid.

% Eero Simoncelli, 6/96.

function res =  spyrHigh(pyr,pind)

res  = pyrBand(pyr, pind, 1);

