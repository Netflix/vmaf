% [VEC] = columnize(MTX)
% 
% Pack elements of MTX into a column vector.  Just provides a
% function-call notatoin for the operation MTX(:)

function vec = columnize(mtx)

vec = mtx(:);
