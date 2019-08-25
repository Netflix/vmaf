% [RES] = ifftshift (MTX)
%
% Inverse of MatLab's FFTSHIFT.  That is,
%     ifftshift(fftshift(MTX)) = MTX
%   for any size MTX.

% Eero Simoncelli, 2/97.

function [res]  = ifftshift(mtx)

sz = size(mtx);
DC = ceil((sz+1)./2);			% location of DC term in a matlab fft.

res = [mtx(DC(1):sz(1), DC(2):sz(2)) , mtx(DC(1):sz(1), 1:DC(2)-1); ...
       mtx(1:DC(1)-1, DC(2):sz(2)) , mtx(1:DC(1)-1, 1:DC(2)-1)];
