% [N,X] = histo(MTX, nbinsOrBinsize, binCenter);
%
% Compute a histogram of (all) elements of MTX.  N contains the histogram
% counts, X is a vector containg the centers of the histogram bins.
%
% nbinsOrBinsize (optional, default = 101) specifies either
% the number of histogram bins, or the negative of the binsize.
%
% binCenter (optional, default = mean2(MTX)) specifies a center position
% for (any one of) the histogram bins.
%
% How does this differ from MatLab's HIST function?  This function:
%   - allows uniformly spaced bins only.
%   +/- operates on all elements of MTX, instead of columnwise.
%   + is much faster (approximately a factor of 80 on my machine).
%   + allows specification of number of bins OR binsize.  Default=101 bins.
%   + allows (optional) specification of binCenter.

% Eero Simoncelli, 3/97.

function [N, X] = histo(mtx, nbins, binCtr)

%% NOTE: THIS CODE IS NOT ACTUALLY USED! (MEX FILE IS CALLED INSTEAD)

fprintf(1,'WARNING: You should compile the MEX version of "histo.c",\n         found in the MEX subdirectory of matlabPyrTools, and put it in your matlab path.  It is MUCH faster.\n');

mtx = mtx(:);

%------------------------------------------------------------
%% OPTIONAL ARGS:

[mn,mx] = range2(mtx);

if (exist('binCtr') ~= 1) 
  binCtr =  mean(mtx);
end

if (exist('nbins') == 1) 
  if (nbins < 0)
    binSize = -nbins;
  else
    binSize = ((mx-mn)/nbins);
    tmpNbins = round((mx-binCtr)/binSize) - round((mn-binCtr)/binSize);
    if (tmpNbins ~= nbins)
      warning('Using %d bins instead of requested number (%d)',tmpNbins,nbins);
    end
  end
else
  binSize = ((mx-mn)/101);
end

firstBin = binCtr + binSize*round( (mn-binCtr)/binSize );

tmpNbins = round((mx-binCtr)/binSize) - round((mn-binCtr)/binSize);

bins = firstBin + binSize*[0:tmpNbins];

[N, X] = hist(mtx, bins);
