% RANGE = showIm (MATRIX, RANGE, ZOOM, LABEL, NSHADES )
% 
% Display a MatLab MATRIX as a grayscale image in the current figure,
% inside the current axes.  If MATRIX is complex, the real and imaginary 
% parts are shown side-by-side, with the same grayscale mapping.
% 
% If MATRIX is a string, it should be the name of a variable bound to a 
% MATRIX in the base (global) environment.  This matrix is displayed as an 
% image, with the title set to the string.
% 
% RANGE (optional) is a 2-vector specifying the values that map to
% black and white, respectively.  Passing a value of 'auto' (default)
% sets RANGE=[min,max] (as in MatLab's imagesc).  'auto2' sets
% RANGE=[mean-2*stdev, mean+2*stdev].  'auto3' sets
% RANGE=[p1-(p2-p1)/8, p2+(p2-p1)/8], where p1 is the 10th percentile
% value of the sorted MATRIX samples, and p2 is the 90th percentile
% value.
% 
% ZOOM specifies the number of matrix samples per screen pixel.  It
% will be rounded to an integer, or 1 divided by an integer.  A value
% of 'same' or 'auto' (default) causes the zoom value to be chosen
% automatically to fit the image into the current axes.  A value of
% 'full' fills the axis region (leaving no room for labels).  See
% pixelAxes.m.
% 
% If LABEL (optional, default = 1, unless zoom='full') is non-zero, the range 
% of values that are mapped into the gray colormap and the dimensions 
% (size) of the matrix and zoom factor are printed below the image.  If label 
% is a string, it is used as a title.
% 
% NSHADES (optional) specifies the number of gray shades, and defaults
% to the size of the current colormap.

% Eero Simoncelli, 6/96.

%%TODO: should use "newplot"

function range = showIm( im, range, zoom, label, nshades );

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (nargin < 1)
  error('Requires at least one input argument.'); 
end

MLv = version;

if isstr(im)
  if (strcmp(MLv(1),'4'))
    error('Cannot pass string arg for MATRIX in MatLab version 4.x');
  end
  label = im;
  im = evalin('base',im);
end

if (exist('range') ~= 1)
  range = 'auto1';
end

if (exist('nshades') ~= 1)
  nshades = size(colormap,1);
end
nshades = max( nshades, 2 );

if (exist('zoom') ~= 1)
  zoom = 'auto';
end

if (exist('label') ~= 1)
  if strcmp(zoom,'full')
    label = 0;				% no labeling
  else					
    label = 1;				% just print grayrange & dims
  end
end

%------------------------------------------------------------

%% Automatic range calculation: 
if (strcmp(range,'auto1') | strcmp(range,'auto'))
  if isreal(im)
    [mn,mx] = range2(im);
  else
    [mn1,mx1] = range2(real(im));
    [mn2,mx2] =  range2(imag(im));
    mn = min(mn1,mn2);
    mx = max(mx1,mx2);
  end
  if any(size(im)==1)
    pad = (mx-mn)/12;			% MAGIC NUMBER: graph padding
    range = [mn-pad, mx+pad];
  else
    range = [mn,mx];
  end

elseif strcmp(range,'auto2')
  if isreal(im)
    stdev = sqrt(var2(im));
    av = mean2(im);
  else
    stdev = sqrt((var2(real(im)) + var2(imag(im)))/2);
    av = (mean2(real(im)) + mean2(imag(im)))/2;
  end
  range = [av-2*stdev,av+2*stdev]; 	% MAGIC NUMBER: 2 stdevs

elseif strcmp(range, 'auto3')
  percentile = 0.1;			% MAGIC NUMBER: 0<p<0.5
  [N,X] = histo(im);
  binsz = X(2)-X(1);
  N = N+1e-10;  % Ensure cumsum will be monotonic for call to interp1
  cumN = [0, cumsum(N)]/sum(N);
  cumX = [X(1)-binsz, X] + (binsz/2);
  ctrRange = interp1(cumN,cumX, [percentile, 1-percentile]);
  range = mean(ctrRange) + (ctrRange-mean(ctrRange))/(1-2*percentile);

elseif isstr(range)
  error(sprintf('Bad RANGE argument: %s',range))

end

if ((range(2) - range(1)) <= eps)
  range(1) = range(1) - 0.5;
  range(2) = range(2) + 0.5;
end


if isreal(im)
  factor=1;
else
  factor = 1+sqrt(-1);
end

xlbl_offset = 0; % default value

if (~any(size(im)==1))
  %% MatLab's "image" rounds when mapping to the colormap, so we compute
  %%      (im-r1)*(nshades-1)/(r2-r1) + 1.5 
  mult = ((nshades-1) / (range(2)-range(1)));
  d_im = (mult * im) + factor*(1.5 - range(1)*mult);
end

if isreal(im)
  if (any(size(im)==1))
    hh = plot( im);
    axis([1, prod(size(im)), range]);
  else
    hh = image( d_im );
    axis('off');
    zoom = pixelAxes(size(d_im),zoom);
  end
else
  if (any(size(im)==1))
    subplot(2,1,1);
    hh = plot(real(im));
    axis([1, prod(size(im)), range]);
    subplot(2,1,2);
    hh = plot(imag(im));
    axis([1, prod(size(im)), range]);
  else
    subplot(1,2,1);
    hh = image(real(d_im));
    axis('off'); zoom = pixelAxes(size(d_im),zoom);
    ax = gca; orig_units = get(ax,'Units');
    set(ax,'Units','points');
    pos1 = get(ax,'Position');
    set(ax,'Units',orig_units);
    subplot(1,2,2);
    hh = image(imag(d_im));
    axis('off'); zoom = pixelAxes(size(d_im),zoom);
    ax = gca; orig_units = get(ax,'Units');
    set(ax,'Units','points');
    pos2 = get(ax,'Position');
    set(ax,'Units',orig_units);
    xlbl_offset = (pos1(1)-pos2(1))/2;
  end
end  

if ~any(size(im)==1)
  colormap(gray(nshades));
end

if ((label ~= 0))
  if isstr(label)
    title(label);
    h = get(gca,'Title');
    orig_units = get(h,'Units');
    set(h,'Units','points');
    pos = get(h,'Position');
    pos(1:2) = pos(1:2) + [xlbl_offset, -3]; % MAGIC NUMBER: y pixel offset
    set(h,'Position',pos);
    set(h,'Units',orig_units);
  end

  if (~any(size(im)==1))
    if (zoom > 1)
      zformat = sprintf('* %d',round(zoom));
    else
      zformat = sprintf('/ %d',round(1/zoom));
    end
    if isreal(im) 
      format=[' Range: [%.3g, %.3g] \n Dims: [%d, %d] ', zformat];
        else
      format=['Range: [%.3g, %.3g]  ----  Dims: [%d, %d]', zformat];
    end
    xlabel(sprintf(format, range(1), range(2), size(im,1), size(im,2)));
    h = get(gca,'Xlabel');
    set(h,'FontSize', 9); 		% MAGIC NUMBER: font size!!!

    orig_units = get(h,'Units');
    set(h,'Units','points');  
    pos = get(h,'Position');
    pos(1:2) = pos(1:2) + [xlbl_offset, 10]; % MAGIC NUMBER: y offset in points
    set(h,'Position',pos);
    set(h,'Units',orig_units);

    set(h,'Visible','on');		% axis('image') turned the  xlabel  off...
  end
end

return;
