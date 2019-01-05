function run_speed(ref_filename, dis_filename, width, height, bands, yuv_type)
%RUN_SPEED Summary of this function goes here
%   Detailed explanation goes here

path(path,'./SpEED');
calcSpEEDScore(ref_filename, dis_filename, width, height, bands, yuv_type)
end




