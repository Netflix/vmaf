% [RES] = shift(MTX, OFFSET)
% 
% Circular shift 2D matrix samples by OFFSET (a [Y,X] 2-vector),
% such that  RES(POS) = MTX(POS-OFFSET).

function res = shift(mtx, offset)

dims = size(mtx);

offset = mod(-offset,dims);

res = [ mtx(offset(1)+1:dims(1), offset(2)+1:dims(2)),  ...
          mtx(offset(1)+1:dims(1), 1:offset(2));        ...
        mtx(1:offset(1), offset(2)+1:dims(2)),          ...
	  mtx(1:offset(1), 1:offset(2)) ];
