function [y, cb, cr] = readframefromfid_all_fmts(fid,height,width,fmt)

% works for both 8bit and 10bit
if strcmp(fmt, 'yuv422p')
    y = fread(fid,width*height, 'uchar')';
    cb = fread(fid,width*height/2, 'uchar')';
    cr = fread(fid,width*height/2, 'uchar')';
elseif strcmp(fmt, 'yuv444p')
    y = fread(fid,width*height, 'uchar')';
    cb = fread(fid,width*height, 'uchar')';
    cr = fread(fid,width*height, 'uchar')';
elseif strcmp(fmt, 'yuv420p')
    y = fread(fid,width*height, 'uchar')';
    cb = fread(fid,width*height/4, 'uchar')';
    cr = fread(fid,width*height/4, 'uchar')';
elseif strcmp(fmt, 'yuv420p10le')
    y = fread(fid,width*height, 'uint16')';
    cb = fread(fid,width*height/4, 'uint16')';
    cr = fread(fid,width*height/4, 'uint16')';
    y = y / 4.0;
    cb = cb / 4.0;
    cr = cr / 4.0;
elseif strcmp(fmt, 'yuv444p10le')
    y = fread(fid,width*height, 'uchar')';
    cb = fread(fid,width*height, 'uchar')';
    cr = fread(fid,width*height, 'uchar')';
    y = y / 4.0;
    cb = cb / 4.0;
    cr = cr / 4.0;
else
    y = fread(fid,width*height, 'uchar')';
    cb = fread(fid,width*height/2, 'uchar')';
    cr = fread(fid,width*height/2, 'uchar')';
    y = y / 4.0;
    cb = cb / 4.0;
    cr = cr / 4.0;
end