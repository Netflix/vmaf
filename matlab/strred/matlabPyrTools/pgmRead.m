% IM = pgmRead( FILENAME )
%
% Load a pgm image into a MatLab matrix.  
%   This format is accessible from the XV image browsing utility.
%   Only works for 8bit gray images (raw or ascii)

% Hany Farid, Spring '96.  Modified by Eero Simoncelli, 6/96.

function im = pgmRead( fname );

[fid,msg] = fopen( fname, 'r' );

if (fid == -1)
  error(msg);
end

%%% First line contains ID string:
%%% "P1" = ascii bitmap, "P2" = ascii greymap,
%%% "P3" = ascii pixmap, "P4" = raw bitmap, 
%%% "P5" = raw greymap, "P6" = raw pixmap
TheLine = fgetl(fid);
format  = TheLine;		

if ~((format(1:2) == 'P2') | (format(1:2) == 'P5'))
  error('PGM file must be of type P2 or P5');
end

%%% Any number of comment lines
TheLine  = fgetl(fid);
while TheLine(1) == '#' 
	TheLine = fgetl(fid);
end

%%% dimensions
sz = sscanf(TheLine,'%d',2);
xdim = sz(1);
ydim = sz(2);
sz = xdim * ydim;

%%% Maximum pixel value
TheLine  = fgetl(fid);
maxval = sscanf(TheLine, '%d',1);

%%im  = zeros(dim,1);
if (format(2) == '2')
  [im,count]  = fscanf(fid,'%d',sz);
else
  [im,count]  = fread(fid,sz,'uchar');
end

fclose(fid);

if (count == sz)
  im = reshape( im, xdim, ydim )';
else
  fprintf(1,'Warning: File ended early!');
  im = reshape( [im ; zeros(sz-count,1)], xdim, ydim)';
end
	  
