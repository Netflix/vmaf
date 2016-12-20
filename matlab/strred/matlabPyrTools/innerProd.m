% RES = innerProd(MTX)
%
% Compute (MTX' * MTX) efficiently (i.e., without copying the matrix)

function res = innerProd(mtx)

fprintf(1,['WARNING: You should compile the MEX version of' ...
	   ' "innerProd.c",\n         found in the MEX subdirectory' ...
	   ' of matlabPyrTools, and put it in your matlab path.' ...
	   ' It is MUCH faster and requires less memory.\n']);

res = mtx' * mtx;
