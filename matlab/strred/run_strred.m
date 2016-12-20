function run_strred(ref_filename, dis_filename, nframes, width, height)

path(path,'./strred');
path(path,'./matlabPyrTools');

[rreds, rredt] = calcStrredScore(ref_filename, dis_filename, nframes, height, width)

% dmos_score = rred2dmos(strred_score)

end
























