% [ZOOM] = pixelAxes(DIMS, ZOOM)
%
% Set the axes of the current plot to cover a multiple of DIMS pixels,
% thereby eliminating screen aliasing artifacts when displaying an
% image of size DIMS.  
% 
% ZOOM (optional, default='same') expresses the desired number of
% samples displayed per screen pixel.  It should be a scalar, which
% will be rounded to the nearest integer, or 1 over an integer.  It
% may also be the string 'same' or 'auto', in which case the value is chosen so
% as to produce an image closest in size to the currently displayed
% image.  It may also be the string 'full', in which case the image is
% made as large as possible while still fitting in the window.

% Eero Simoncelli, 2/97.

function [zoom] = pixelAxes(dims, zoom)

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (exist('zoom') ~= 1)
  zoom = 'same';
end

%% Reverse  dimension order, since Figure Positions reported as (x,y).
dims = dims(2:-1:1);

%% Use MatLab's axis function to force square pixels, etc:
axis('image');
ax = gca;

oldunits = get(ax,'Units');

if strcmp(zoom,'full');
  set(ax,'Units','normalized');
  set(ax,'Position',[0 0 1 1]);
  zoom = 'same';
end

set(ax,'Units','pixels');
pos = get(ax,'Position');
ctr = pos(1:2)+pos(3:4)/2;

if (strcmp(zoom,'same') | strcmp(zoom,'auto'))
  %% HACK: enlarge slightly so that floor doesn't round down 
  zoom = min( pos(3:4) ./ (dims - 1) );
elseif isstr(zoom)
  error(sprintf('Bad ZOOM argument: %s',zoom));
end

%% Force zoom value to be an integer, or inverse integer.
if (zoom < 0.75)
  zoom = 1/ceil(1/zoom);
  %% Round upward, subtracting 0.5 to avoid floating point errors.
  newsz = ceil(zoom*(dims-0.5));
else
  zoom = floor(zoom + 0.001);		% Avoid floating pt errors
  if (zoom < 1.5)			% zoom=1
    zoom = 1;
    newsz = dims + 0.5;
  else
    newsz = zoom*(dims-1) + mod(zoom,2);
  end
end 

set(ax,'Position', [floor(ctr-newsz/2)+0.5, newsz] )

% Restore units
set(ax,'Units',oldunits);
