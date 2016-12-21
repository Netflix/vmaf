% [RES] = clip(IM, MINVALorRANGE, MAXVAL)
%
% Clip values of matrix IM to lie between minVal and maxVal:
%      RES = max(min(IM,MAXVAL),MINVAL)
% The first argument can also specify both min and max, as a 2-vector.
% If only one argument is passed, the range defaults to [0,1].

function res = clip(im, minValOrRange, maxVal)

if (exist('minValOrRange') ~= 1) 
  minVal = 0; 
  maxVal = 1;
elseif (length(minValOrRange) == 2)
  minVal = minValOrRange(1);
  maxVal = minValOrRange(2);
elseif (length(minValOrRange) == 1)
  minVal = minValOrRange;
  if (exist('maxVal') ~= 1)
    maxVal=minVal+1;
  end
else
  error('MINVAL must be  a scalar or a 2-vector');
end

if ( maxVal < minVal )
  error('MAXVAL should be less than MINVAL');
end

res = im;
res(find(im < minVal)) = minVal;
res(find(im > maxVal)) = maxVal;

