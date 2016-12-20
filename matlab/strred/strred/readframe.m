function [y,cb,cr]=readframe(vidfilename,framenum,height,width)
%
% [y1, y2]=vqeg_readframe(vidfilename, framenum, format)
% format = 625 or 525
% returns the luminance fields y1 and y2 for the framenum frame of the vqeg video sequqnce vidfilename
% y1 is the temporally earlier field, and y2 is the temporally later

% height=432;
% width=768;

fid=fopen(vidfilename);

fseek(fid, (framenum-1)*width*height*1.5, 'bof');
%fread(fid, (framenum-1)*width*height*1.5, 'uchar');

y=fread(fid,width*height, 'uchar')';
cb=fread(fid,width*height/4, 'uchar')';
cr=fread(fid,width*height/4, 'uchar')';
fclose (fid);
