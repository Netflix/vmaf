function [y,cb,cr]=readframefromfid(fid,height,width)

% for now, read yuv420p only

y=fread(fid,width*height, 'uchar')';
cb=fread(fid,width*height/4, 'uchar')';
cr=fread(fid,width*height/4, 'uchar')';

end