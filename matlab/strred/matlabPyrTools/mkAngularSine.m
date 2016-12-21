% IM = mkAngularSine(SIZE, HARMONIC, AMPL, PHASE, ORIGIN)
%
% Make an angular sinusoidal image:
%     AMPL * sin( HARMONIC*theta + PHASE),
% where theta is the angle about the origin.
% SIZE specifies the matrix size, as for zeros().  
% AMPL (default = 1) and PHASE (default = 0) are optional.

% Eero Simoncelli, 2/97.

function [res] = mkAngularSine(sz, harmonic, ampl, ph, origin)

sz = sz(:);
if (size(sz,1) == 1)
  sz = [sz,sz];
end

mxsz = max(sz(1),sz(2));

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('harmonic') ~= 1)
  harmonic = 1;
end

if (exist('ampl') ~= 1)
  ampl = 1;
end

if (exist('ph') ~= 1)
  ph = 0;
end

if (exist('origin') ~= 1)
  origin = (sz+1)/2;
end

%------------------------------------------------------------

res = ampl * sin(harmonic*mkAngle(sz,ph,origin) + ph);

