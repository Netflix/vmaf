function result = resize(orig, newSize, align, padding)
% result = resize(orig, newSize, align, padding)
%
% if newSize is larger than orig size, pad with padding;
% if newSize is smaller than orig size, truncate to fit. 
% align specifies alignment. 0=centered
%                           -1=left (up) aligned
%                            1=right (down) aligned
% For example, align=[0 -1] centers on rows (y) and left align on columns (x).
%              align=1 aligns left on columns and top on rows.
% align defaults to 0, and padding defaults to 0.
%
% Xuemei Zhang
% Last Modified 8/21/96

if (nargin<3)
  align = [0 0];
end 
if (nargin<4)
  padding = 0;
end

if (length(newSize)==1)
  newSize = [newSize newSize];
end
if (length(align)==1)
  align = [align align];
end

[m1,n1] = size(orig);
m2 = newSize(1);
n2 = newSize(2);
m = min(m1, m2);
n = min(n1, n2);

result = ones(m2, n2) * padding;

start1 = [floor((m1-m)/2*(1+align(1))) floor((n1-n)/2*(1+align(2)))] + 1;
start2 = [floor((m2-m)/2*(1+align(1))) floor((n2-n)/2*(1+align(2)))] + 1;

result(start2(1):(start2(1)+m-1), start2(2):(start2(2)+n-1)) = ...
   orig(start1(1):(start1(1)+m-1), start1(2):(start1(2)+n-1));
