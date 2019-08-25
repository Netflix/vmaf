% IM = mkZonePlate(SIZE, AMPL, PHASE)
%
% Make a "zone plate" image:
%     AMPL * cos( r^2 + PHASE)
% SIZE specifies the matrix size, as for zeros().  
% AMPL (default = 1) and PHASE (default = 0) are optional.

% Eero Simoncelli, 6/96.

function [res] = mkZonePlate(sz, ampl, ph)

sz = sz(:);
if (size(sz,1) == 1)
  sz = [sz,sz];
end

mxsz = max(sz(1),sz(2));

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('ampl') ~= 1)
  ampl = 1;
end

if (exist('ph') ~= 1)
  ph = 0;
end

%------------------------------------------------------------

res = ampl * cos( (pi/mxsz) * mkR(sz,2) + ph );

