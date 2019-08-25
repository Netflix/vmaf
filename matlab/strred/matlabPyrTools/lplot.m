% lplot(VEC, XRANGE)
%
% Plot VEC, a vector, in  "lollipop" format.  
% XRANGE (optional, default = [1,length(VEC)]), should be a 2-vector 
% specifying the X positions (for labeling purposes) of the first and 
% last sample of VEC.

% Mark Liberman, Linguistics Dept, UPenn, 1994.

function lplot(x,xrange)

if (exist('xrange') ~= 1)
  xrange = [1,length(x)];
end

msize = size(x);
if ( msize(2) == 1)
  x = x';
elseif (msize(1) ~= 1)
  error('First arg must be a vector');
end

if (~isreal(x))
  fprintf(1,'Warning: Imaginary part of signal ignored\n');
  x = abs(x);
end

N = length(x);
index = xrange(1) + (xrange(2)-xrange(1))*[0:(N-1)]/(N-1)
xinc = index(2)-index(1);

xx = [zeros(1,N);x;zeros(1,N)];
indexis = [index;index;index];
xdiscrete = [0 xx(:)' 0];
idiscrete = [index(1)-xinc indexis(:)' index(N)+xinc];

[mn,mx] = range2(xdiscrete);
ypad = (mx-mn)/12;			% MAGIC NUMBER: graph padding

plot(idiscrete, xdiscrete, index, x, 'o');
axis([index(1)-xinc, index(N)+xinc, mn-ypad, mx+ypad]);

return
