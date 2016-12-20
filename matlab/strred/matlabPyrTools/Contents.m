% Image and Multi-scale Pyramid Tools
% Version 1.3,  October 2004.
% Created: Early Spring, 1996. Eero Simoncelli, eero.simoncelli@nyu.edu
%
% See README file for brief description.
% See ChangeLog file for latest modifications. 
% See TUTORIALS subdirectory for demonstrations.
% Type "help <command-name>" for documentation on individual commands.
% -----------------------------------------------------------------
% Synthetic Images (matrices):
%   mkImpulse  - Make an image containing an impulse.
%   mkRamp     - Make an image containing a ramp function.
%   mkR        - Make an image containing distance from the origin.
%   mkAngle    - Make an image containing angle about origin.
%   mkDisc     - Make an image containing a disk image.
%   mkGaussian - Make an image containing a Gaussian function.
%   mkZonePlate - Make an image containing a zone plate (cos(r^2)).
%   mkAngularSine - Make an image containing an angular sine wave (pinwheel).
%   mkSine     - Make an image containing a sine grating.
%   mkSquare   - Make an image containing a square grating.
%   mkFract    - Make an image containing fractal (1/f) noise.
%
% Point Operations:
%   clip       - clip values to a range. 
%   pointOp    - Lookup table (much faster than interp1) [MEX file]
%   histo      - Efficient histogram computation [MEX file]
%   histoMatch - Modify matrix elements to match specified histogram stats.
%
% Convolution (first two are significantly faster):
%   corrDn     - Correlate & downsample with boundary-handling [MEX file]
%   upConv     - Upsample & convolve with boundary-handling [MEX file]
%   blurDn     - Blur and subsample a signal/image.
%   upBlur     - Upsample and blur a signal/image.
%   blur       - Multi-scale blurring, calls blurDn and then upBlur.
%   cconv2     - Circular convolution.
%   rconv2     - Convolution with reflected boundaries.
%   zconv2     - Convolution assuming zeros beyond image boundaries.
%
% General pyramids:
%   pyrLow     - Access lowpass subband from (any type of) pyramid
%   pyrBand    - Access a subband from (any type of) pyramid
%   setPyrBand - Insert an image into (any type of) pyramid as a subband 
%   pyrBandIndices - Returns indices for given band in a pyramid vector
%   maxPyrHt   - compute maximum number of scales in a pyramid
%
% Gaussian/Laplacian Pyramids:
%   buildGpyr  - Build a Gaussian pyramid of an input signal/image.
%   buildLpyr  - Build a Laplacian pyramid of an input signal/image.
%   reconLpyr  - Reconstruct (invert) the Laplacian pyramid transform.
%
% Separable orthonormal QMF/wavelet Pyramids:
%   buildWpyr  - Build a separable wavelet representation of an input signal/image.
%   reconWpyr  - Reconstruct (invert) the wavelet transform.
%   wpyrBand   - Extract a single band of the wavelet representation.
%   wpyrLev    - Extract (packed) subbands at a particular level
%   wpyrHt     - Number of levels (height) of a wavelet pyramid.
%
% Steerable Pyramids:
%   buildSpyr  - Build a steerable pyramid representation of an input image.
%   reconSpyr  - Reconstruct (invert) the steerable pyramid transform.
%   buildSFpyr - Build a steerable pyramid representation in the Fourier domain.
%   reconSFpyr - Reconstruct (invert) the (Fourier domain) steerable pyramid transform.
%   spyrBand   - Extract a single band from a steerable pyramid.
%   spyrHigh   - Highpass residual band.
%   spyrLev    - A whole level (i.e., all images at a given scale) of a steerable pyramid.
%   spyrHt     - Number of levels (height) of a steerable pyramid.
%   spyrNumBands - Number of orientation bands in a steerable pyramid.
%
% Steerable filters / derivatives:
%   imGradient - Compute gradient of image using directionally accurete filters.
%   steer      - Steer filters (or responses).
%   steer2HarmMtx - Construct a matrix mapping direcional basis to angular harmonics. 
% 
% Filters:
%   binomialFilter  - returns a filter of binomial coefficients.
%   namedFilter     - some typical Laplacian/Wavelet pyramid filters
%   spNFilters      - Set of Nth order steerable pyramid filters.
%   derivNFiltersS  - Matched set of S-tap 1D derivatives, orders 0 to N.
% 
% Display:
%   showIm     - Display a matrix (real or complex) as grayscale image(s).
%                Displays dimensions, subsampling, and range of pixel values.
%   showLpyr   - Display a Laplacian pyramid.
%   showWpyr   - Display a separable wavelet pyramid.
%   showSpyr   - Display a steerable pyramid.
%   lplot      - "lollipop" plot.
%   nextFig    - Make next figure window current.
%   pixelAxes  - Make image display use an integer number of pixels 
%                per sample to avoid resampling artifacts.
% 
% Statistics (for 2D Matrices):
%   range2     - Min and max of image (matrix) [MEX file]
%   mean2      - Sample mean of an image (matrix). 
%   var2       - Sample variance of an image (matrix). 
%   skew2      - Sample skew (3rd moment / variance^1.5) of an image (matrix). 
%   kurt2      - Sample kurtosis (4th moment / variance^2) of an image (matrix). 
%   entropy2   - Sample entropy of an image (matrix).
%   imStats    - Report sample statistics of an image, or pair of images.
%
% Miscellaneous:
%   pgmRead    - Load a "pgm" image into a MatLab matrix [try einstein.pgm,feynman.pgm]
%   pgmWrite   - Write a MatLab matrix to a "pgm" image file.
%   shift      - circular shift a 2D matrix by an arbitrary amount.
%   vectify    - pack matrix into column vector (i.e., function to compute mtx(:)).
%   ifftshift  - inverse of MatLab's FFTSHIFT (differs for odd-length dimensions)
%   rcosFn     - return a lookup table of a raised-cosine threshold fn.
%   innerProd  - Compute M'*M efficiently (i.e., do not copy) [MEX file]
