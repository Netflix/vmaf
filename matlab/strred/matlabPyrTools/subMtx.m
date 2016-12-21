% MTX = subMtx(VEC, DIMENSIONS, START_INDEX)
%
% Reshape a portion of VEC starting from START_INDEX (optional,
% default=1) to the given dimensions.

% Eero Simoncelli, 6/96.

function mtx = subMtx(vec, sz, offset)

if (exist('offset') ~= 1)
   offset = 1;
end

vec = vec(:);
sz = sz(:);

if (size(sz,1) ~= 2)
  error('DIMENSIONS must be a 2-vector.');
end

mtx = reshape( vec(offset:offset+prod(sz)-1), sz(1), sz(2) );
