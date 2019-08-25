% [X, Y] = rcosFn(WIDTH, POSITION, VALUES)
%
% Return a lookup table (suitable for use by INTERP1) 
% containing a "raised cosine" soft threshold function:
% 
%    Y =  VALUES(1) + (VALUES(2)-VALUES(1)) *
%              cos^2( PI/2 * (X - POSITION + WIDTH)/WIDTH )
%
% WIDTH is the width of the region over which the transition occurs
% (default = 1). POSITION is the location of the center of the
% threshold (default = 0).  VALUES (default = [0,1]) specifies the
% values to the left and right of the transition.

% Eero Simoncelli, 7/96.

function [X, Y] = rcosFn(width,position,values)

%------------------------------------------------------------
% OPTIONAL ARGS:

if (exist('width') ~= 1)
  width = 1;
end

if (exist('position') ~= 1)
  position = 0;
end

if (exist('values') ~= 1)
  values = [0,1];
end

%------------------------------------------------------------

sz = 256;  %% arbitrary!

X    = pi * [-sz-1:1] / (2*sz);

Y = values(1) + (values(2)-values(1)) * cos(X).^2;

%    Make sure end values are repeated, for extrapolation...
Y(1) = Y(2);
Y(sz+3) = Y(sz+2);

X = position + (2*width/pi) * (X + pi/4);
