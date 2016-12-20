% E = ENTROPY2(MTX,BINSIZE) 
% 
% Compute the first-order sample entropy of MTX.  Samples of VEC are
% first discretized.  Optional argument BINSIZE controls the
% discretization, and defaults to 256/(max(VEC)-min(VEC)).
%
% NOTE: This is a heavily  biased estimate of entropy when you
% don't have much data.

% Eero Simoncelli, 6/96.

function res = entropy2(mtx,binsize)

%% Ensure it's a vector, not a matrix.
vec = mtx(:);
[mn,mx] = range2(vec);

if (exist('binsize') == 1)
  nbins = max((mx-mn)/binsize, 1);
else
  nbins = 256;
end
  
[bincount,bins] = histo(vec,nbins);

%% Collect non-zero bins:
H = bincount(find(bincount));
H = H/sum(H);

res = -sum(H .* log2(H));

