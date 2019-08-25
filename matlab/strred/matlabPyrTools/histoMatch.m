% RES = histoMatch(MTX, N, X)
%
% Modify elements of MTX so that normalized histogram matches that
% specified by vectors X and N, where N contains the histogram counts
% and X the histogram bin positions (see histo).

% Eero Simoncelli, 7/96.

function res = histoMatch(mtx, N, X)

if ( exist('histo') == 3 )
  [oN, oX] = histo(mtx(:), size(X(:),1));
else
  [oN, oX] = hist(mtx(:), size(X(:),1));
end

oStep = oX(2) - oX(1);
oC = [0, cumsum(oN)]/sum(oN);
oX = [oX(1)-oStep/2, oX+oStep/2];

N = N(:)';
X = X(:)';
N = N + mean(N)/(1e8);   %% HACK: no empty bins ensures nC strictly monotonic

nStep = X(2) - X(1);
nC = [0, cumsum(N)]/sum(N);
nX = [X(1)-nStep/2, X+nStep/2];

nnX = interp1(nC, nX, oC, 'linear');

if ( exist('pointOp') == 3 )
  res = pointOp(mtx, nnX, oX(1), oStep);
else
  res = reshape(interp1(oX, nnX, mtx(:)),size(mtx,1),size(mtx,2));
end
