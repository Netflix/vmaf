% %-----------------------------------------------------
function [I mp gabors] = lo_index( ref, dst )
% Calculate the low quality index by calculating the Gabor analysis and
% then the local statistics,
% Author: Eric Larson
% Department of Electrical and Computer Engineering
% Oklahoma State University, 2008
% Image Coding and analysis lab


% Decompose using Gabor Analysis
gabRef  = gaborconvolve( double( ref ) );
gabDst  = gaborconvolve( double( dst ) );

[M N]   = size( gabRef{1,1} );
O       = size( gabRef, 1 ) * size( gabRef, 2 );
s       = [0.5 0.75 1 5 6];
mp      = zeros( M, N );
s       = s./sum(s(:));


if( nargout == 3)
  gabors.dst = zeros( M, N, 5*4*3 );
  gabors.ref = zeros( M, N, 5*4*3 );
end

im_k = 0;
BSIZE = 16;
for gb_i = 1:5
  for gb_j = 1:4
    
    im_k  = im_k + 1;
%     fprintf('%d ',im_k)
    
    if( nargout == 3 )
      [gabors.ref(:,:,3*(im_k-1)+1), gabors.ref(:,:,3*(im_k-1)+2), gabors.ref(:,:,3*(im_k-1)+3)] ...
        = ical_stat( abs( gabRef{gb_i,gb_j} ) );
      
      [gabors.dst(:,:,3*(im_k-1)+1), gabors.dst(:,:,3*(im_k-1)+2), gabors.dst(:,:,3*(im_k-1)+3)] ...
        = ical_stat( abs( gabDst{gb_i,gb_j} ) );
      
    else
      % otherwise keep memory footprint to a minimum
      [std.ref, skw.ref, krt.ref] ...
        = ical_stat( abs( gabRef{gb_i,gb_j} ) );
      [std.dst, skw.dst, krt.dst] ...
        = ical_stat( abs( gabDst{gb_i,gb_j} ) );
      
      mp = mp + s(gb_i) .* ( abs( std.ref - std.dst )... std
        + 2.*abs( skw.ref - skw.dst )...              skew
        +    abs( krt.ref - krt.dst ) ); %        kurtosis
    end
  end
end

% fprintf('\n ')

% kill the edges
mp2 = mp( BSIZE+1:end-BSIZE-1, BSIZE+1:end-BSIZE-1);

I = norm( mp2 (:), 2 ) / sqrt( length( mp2(:) ) ) ;
% = norm of statistical diff map / ( root image size * normalization )
end


%-----------------------------------------------------

function EO = gaborconvolve( im )

% GABORCONVOLVE - function for convolving image with log-Gabor filters
%
%   Usage: EO = gaborconvolve(im,  nscale, norient )
%
% For details of log-Gabor filters see:
% D. J. Field, "Relations Between the Statistics of Natural Images and the
% Response Properties of Cortical Cells", Journal of The Optical Society of
% America A, Vol 4, No. 12, December 1987. pp 2379-2394
% Notes on filter settings to obtain even coverage of the spectrum
% dthetaOnSigma 1.5
% sigmaOnf  .85   mult 1.3
% sigmaOnf  .75   mult 1.6     (bandwidth ~1 octave)
% sigmaOnf  .65   mult 2.1
% sigmaOnf  .55   mult 3       (bandwidth ~2 octaves)
%
% Author: Peter Kovesi
% Department of Computer Science & Software Engineering
% The University of Western Australia
% pk@cs.uwa.edu.au  www.cs.uwa.edu.au/~pk
%
% May 2001
% Altered, 2008, Eric Larson
% Altered precomputations, 2011, Eric Larson

nscale          = 5;      %Number of wavelet scales.
norient         = 4;      %Number of filter orientations.
minWaveLength   = 3;      %Wavelength of smallest scale filter.
mult            = 3;      %Scaling factor between successive filters.
sigmaOnf        = 0.55;   %Ratio of the standard deviation of the
%Gaussian describing the log Gabor filter's transfer function
%in the frequency domain to the filter center frequency.
%Orig: 3 6 12 27 64
wavelength      = [minWaveLength ...
  minWaveLength*mult ...
  minWaveLength*mult^2 ...
  minWaveLength*mult^3 ...
  minWaveLength*mult^4 ];

dThetaOnSigma   = 1.5;    %Ratio of angular interval between filter orientations
% 			       and the standard deviation of the angular Gaussian
% 			       function used to construct filters in the
%                              freq. plane.

[rows cols] = size( im );
imagefft    = fft2( im );            % Fourier transform of image

EO = cell( nscale, norient );        % Pre-allocate cell array

% Pre-compute to speed up filter construction
x = ones(rows,1) * (-cols/2 : (cols/2 - 1))/(cols/2);
y = (-rows/2 : (rows/2 - 1))' * ones(1,cols)/(rows/2);
radius = sqrt(x.^2 + y.^2);       % Matrix values contain *normalised* radius from centre.
radius(round(rows/2+1),round(cols/2+1)) = 1; % Get rid of the 0 radius value in the middle
radius = log(radius);
% so that taking the log of the radius will
% not cause trouble.

% Precompute sine and cosine of the polar angle of all pixels about the
% centre point

theta = atan2(-y,x);              % Matrix values contain polar angle.
% (note -ve y is used to give +ve
% anti-clockwise angles)
sintheta = sin(theta);
costheta = cos(theta);
clear x; clear y; clear theta;      % save a little memory

thetaSigma = pi/norient/dThetaOnSigma;  % Calculate the standard deviation of the
% angular Gaussian function used to
% construct filters in the freq. plane.
rows = round(rows/2+1);
cols = round(cols/2+1);
% precompute the scaling filters
logGabors = cell(1,nscale);
for s = 1:nscale                  % For each scale.
    
    % Construct the filter - first calculate the radial filter component.
    fo = 1.0/wavelength(s);                  % Centre frequency of filter.
    rfo = fo/0.5;                         % Normalised radius from centre of frequency plane
    % corresponding to fo.
    tmp = -(2 * log(sigmaOnf)^2);
    tmp2= log(rfo);
    logGabors{s} = exp( (radius-tmp2).^2 /tmp  );

    logGabors{s}( rows, cols ) = 0; % Set the value at the center of the filter
    % back to zero (undo the radius fudge).
end


% The main loop...
for o = 1:norient,                   % For each orientation.
%   fprintf('.');
  angl = (o-1)*pi/norient;           % Calculate filter angle.
  
  % Pre-compute filter data specific to this orientation
  % For each point in the filter matrix calculate the angular distance from the
  % specified filter orientation.  To overcome the angular wrap-around problem
  % sine difference and cosine difference values are first computed and then
  % the atan2 function is used to determine angular distance.
  
  ds = sintheta * cos(angl) - costheta * sin(angl);     % Difference in sine.
  dc = costheta * cos(angl) + sintheta * sin(angl);     % Difference in cosine.
  dtheta = abs(atan2(ds,dc));                           % Absolute angular distance.
  spread = exp((-dtheta.^2) / (2 * thetaSigma^2));      % Calculate the angular filter component.
  
  for s = 1:nscale,                  % For each scale.

    filter = fftshift( logGabors{s} .* spread ); % Multiply by the angular spread to get the filter
    % and swap quadrants to move zero frequency
    % to the corners.
    
    % Do the convolution, back transform, and save the result in EO
    EO{s,o} = ifft2( imagefft .* filter );
    
  end  % ... and process the next scale
  
end  % For each orientation

end

%-----------------------------------------------------