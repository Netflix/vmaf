% [HEIGHT] = lpyrHt(INDICES)
%
% Compute height of Laplacian pyramid with given its INDICES matrix.
% See buildLpyr.m

% Eero Simoncelli, 6/96.

function [ht] =  lpyrHt(pind)

% Don't count lowpass residual band
ht = size(pind,1)-1;
