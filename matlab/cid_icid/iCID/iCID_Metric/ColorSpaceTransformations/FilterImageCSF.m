% Perform constrast-sensitivity filtering of an input image in XYZ
% coordinates. The constrast-sensitivity functions are based on those
% proposed in
%
% E. Reinhard, E. A. Khan, A. O. Akyüz, G. M. Johnson, "Color Imaging:
% Fundamentals and Applications", A K Peters, 2008
%
% for image-difference evaluation with the iCAM image-appearance model.
% However, the luminance CSF was turned into a low-pass filter and all
% values > 1 were clipped to 1.
%
% This code is supplementary material to the article:
%       J. Preiss, F. Fernandes, and P. Urban, "Color-Image Quality 
%       Assessment: From Prediction to Optimization", IEEE Transactions on 
%       Image Processing, pp. 1366-1378, Volume 23, Issue 3, March 2014
%
% Authors:  Jens Preiss, Felipe Fernandes
%           Institute of Printing Science and Technology
%           Technische Universität Darmstadt
%           preiss.science@gmail.com
%           fernandes@idd.tu-darmstadt.de
%           http://www.idd.tu-darmstadt.de/color
%           and
%           Philipp Urban
%           Fraunhofer Institute for Computer Graphics Research IGD
%           philipp.urban@igd.fraunhofer.de
%           http://www.igd.fraunhofer.de/en/Institut/Abteilungen/3DT
%
% Interface:
%           ImageFILT = FilterImageCSF(ImageINP, varargin)
%
% Parameters (mandatory):
%           ImageINP        Input image (in XYZ coordinates)
%
% Parameters (optional):
%           'cpd'         	Cycles per degree of the visual field
%           'wrk_space'     Working color space: 'YCC-RIT' or 'LAB00HL'
%           'pad_image'     Pad image before Fourier transform? Options:
%                           'none': No padding
%                           'post': Pad after last element
%                           'both': Pad before first and after last element
%           'xyz_clip'      Clip XYZ values to the range [0; WP]
%
% Example:
%           ImageFILT = FilterImageCSF(ImageINP, 'cpd', 20);
%
function ImageFILT = FilterImageCSF(ImageINP, varargin)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% INPUT CHECK %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (nargin < 1)
    error('Please specify an input image.');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%% PARAMETER DEFAULTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%

cpd       = 20;         % Cycles per degree of the visual field
wrk_space = 'YCC-RIT';  % Working color space: YCC-RIT (linear) or LAB00HL
pad_image = 'both';     % Pad image before Fourier transform (this is
                        % highly recommended to avoid artifacts at the
                        % image borders, especially at high cpd values);
                        % options are: 'both', 'post', 'none' (see MATLAB
                        % function 'padarray' for details)
xyz_clip  = 0;          % Clip XYZ values to the range [0; WP]

% Check for optional parameters
if (~isempty(varargin))
    for n = 1:2:size(varargin, 2)
        if (strcmp(varargin{n}, 'cpd'))
            cpd = varargin{n+1};
        end
        if (strcmp(varargin{n}, 'wrk_space'))
            wrk_space = varargin{n+1};
        end
        if (strcmp(varargin{n}, 'pad_image'))
            pad_image = varargin{n+1};
        end
        if (strcmp(varargin{n}, 'xyz_clip'))
            xyz_clip = varargin{n+1};
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%% FIXED PARAMETERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% D65/2° white point (required for clipping)
XYZ_W = [0.9505, ...
         1.0000, ...
         1.0888];

% XYZ -> YCC transformation matrix
% (from: "Color Imaging - Fundamentals and Applications")
M_TF = [ 0.0556  0.9981 -0.0254;
         0.9510 -0.9038  0.0000;
         0.0386  1.0822 -1.0276 ];
     
%%%%%%%%%%%%%%%%%% TRANSFORMATION TO WORKING COLOR SPACE %%%%%%%%%%%%%%%%%%

switch (wrk_space)
    case 'YCC-RIT'
        % XYZ -> YCC-RIT
        ImageOPP = ImageXYZ2YCCRIT(ImageINP, M_TF);
    case 'LAB00HL'
        % XYZ -> LAB2000HL
        ImageOPP = ImageXYZ2LAB2000HL(ImageINP);
end

%%%%%%%%%%%%%%%%%%%%%%%%%% CPD-DEPENDENT FILTERS %%%%%%%%%%%%%%%%%%%%%%%%%%

% Get image size
[M,N,D] = size(ImageINP); %#ok<NASGU>

switch (pad_image)
    case 'both'
        Mfilt = 3*M;
        Nfilt = 3*N;
    case 'post'
        Mfilt = 2*M;
        Nfilt = 2*N;
    otherwise
        Mfilt = M;
        Nfilt = N;
end

% Compute quadrant dimensions (the image's spectrum is composed of four
% quadrants - these are the same quadrants that are swapped by MATLAB's
% 'fftshift' method)
% -> Rows
switch (mod(Mfilt, 2))
    case 0  % EVEN number of rows
        Mmin = Mfilt/2;         % Mininum row index (first row)
        Mmax = Mfilt/2 - 1;     % Maximum row index (last  row)
    case 1  % ODD number of rows
        Mmin = (Mfilt-1)/2;     % Mininum row index (first row)
        Mmax = (Mfilt-1)/2;     % Maximum row index (last  row)
end
% -> Cols
switch (mod(Nfilt, 2))
    case 0  % EVEN number of cols
        Nmin = Nfilt/2;         % Minimum col index (first column)
        Nmax = Nfilt/2 - 1;     % Maximum col index (last  column)
    case 1  % ODD number of cols
        Nmin = (Nfilt-1)/2;     % Minimum col index (first column)
        Nmax = (Nfilt-1)/2;     % Maximum col index (last  column)
end

% Create distance matrices (containing distances of individual matrix
% elements form the DC, which is either the center element of the matrix
% (odd rows/cols) or +1/+1 from the center (even rows/cols)).
DistX = ( [Nmin:-1:1  0  1:1:Nmax]' * ones(1,Mfilt) )';
DistY = ( [Mmin:-1:1  0  1:1:Mmax]' * ones(1,Nfilt) );

% Normalize
DistX = DistX./max(DistX(:));
DistY = DistY./max(DistY(:));

% Compute Euclidean distances
Dist = sqrt(DistX.^2 + DistY.^2);

% Compute generic filter matrix
Dist = Dist.*cpd;

% Check if there is no DC amplification for the luminance channel (i.e.,
% the entry of the filter matrix that corresponds to the DC is zero).
% This serves to check if we generated the filter matrix correctly.
Tmp = ifftshift(Dist);
assert(Tmp(1,1) == 0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPUTE FILTERS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LUM_filter = CSF_lum(Dist); % LUM_filter = LUM_filter./max(LUM_filter(:));
CRG_filter = CSF_rg(Dist);
CBY_filter = CSF_by(Dist);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% FILTER IMAGE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Extract luminance channel
LUM = ImageOPP(:,:,1);
CRG = ImageOPP(:,:,2);
CBY = ImageOPP(:,:,3);

% Compute luminance channel mean
LUM_mean = mean2(LUM);

% Luminance channel: subtract mean
LUM = LUM - LUM_mean;

% Pad image before Fourier transform
switch (pad_image)
    case 'both'
        LUM = padarray(LUM, [M N], 'symmetric', 'both');
        CRG = padarray(CRG, [M N], 'symmetric', 'both');
        CBY = padarray(CBY, [M N], 'symmetric', 'both');
    case 'post'
        LUM = padarray(LUM, [M N], 'symmetric', 'post');
        CRG = padarray(CRG, [M N], 'symmetric', 'post');
        CBY = padarray(CBY, [M N], 'symmetric', 'post');
end

% Compute Fourier transforms
LUM_fft2 = fft2(LUM);
CRG_fft2 = fft2(CRG);
CBY_fft2 = fft2(CBY);

% Compute magnitude spectra. Since we subtracted the mean of the luminance
% channel before computing the corresponding Fourier transform, the value
% at the center of the luminance spectrum (the DC) should be close to zero
% now. For the chromatic channels, this value should be quite big.
LUM_fft2 = fftshift(LUM_fft2);
CRG_fft2 = fftshift(CRG_fft2);
CBY_fft2 = fftshift(CBY_fft2);

% Apply filters in the frequency domain
LUM_fft2 = LUM_fft2.*LUM_filter;
CRG_fft2 = CRG_fft2.*CRG_filter;
CBY_fft2 = CBY_fft2.*CBY_filter;

% Inverse FFT shift
LUM_fft2 = ifftshift(LUM_fft2);
CRG_fft2 = ifftshift(CRG_fft2);
CBY_fft2 = ifftshift(CBY_fft2);

% Inverse Fourier transformation
LUM_filt = ifft2(LUM_fft2);
CRG_filt = ifft2(CRG_fft2);
CBY_filt = ifft2(CBY_fft2);

% Undo the padding (cut out the center image)
switch (pad_image)
    case 'both'
        LUM_filt = LUM_filt(M+1:M*2,N+1:N*2);
        CRG_filt = CRG_filt(M+1:M*2,N+1:N*2);
        CBY_filt = CBY_filt(M+1:M*2,N+1:N*2);
    case 'post'
        LUM_filt = LUM_filt(1:M,1:N);
        CRG_filt = CRG_filt(1:M,1:N);
        CBY_filt = CBY_filt(1:M,1:N);
end

% Luminance channel: add mean
LUM_filt = LUM_filt + LUM_mean;

%%%%%%%%%%%%%%%%%%%%%%%% ASSEMBLE RESULTING IMAGE %%%%%%%%%%%%%%%%%%%%%%%%%

% Put together resulting image
ImageFILT(:,:,1) = LUM_filt;
ImageFILT(:,:,2) = CRG_filt;
ImageFILT(:,:,3) = CBY_filt;

%%%%%%%%%%%%%%%%% TRANSFORMATION FROM WORKING COLOR SPACE %%%%%%%%%%%%%%%%%

switch (wrk_space)
    case 'YCC-RIT'
        % YCC-RIT -> XYZ
        ImageFILT = ImageYCCRIT2XYZ(ImageFILT, M_TF);
    case 'LAB00HL'
        % LAB2000HL -> XYZ
        ImageFILT = ImageLAB2000HL2XYZ(ImageFILT);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CLIPPING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if (xyz_clip)
    % 1. Clip all negative XYZ values
    ImageFILT(ImageFILT < 0) = 0;
    
    % 2. Clip all XYZ values greater than the white point (D65/2°)
    D1 = ImageFILT(:,:,1); D1(D1>XYZ_W(1)) = XYZ_W(1); ImageFILT(:,:,1) = D1;
    D2 = ImageFILT(:,:,2); D2(D2>XYZ_W(2)) = XYZ_W(2); ImageFILT(:,:,2) = D2;
    D3 = ImageFILT(:,:,3); D3(D3>XYZ_W(3)) = XYZ_W(3); ImageFILT(:,:,3) = D3;
end

end


% XYZ -> YCC-RIT transformation (for images)
function ImageYCC = ImageXYZ2YCCRIT(ImageXYZ, M_TF)
    [M,N,D] = size(ImageXYZ);
    ImageXYZ = reshape(ImageXYZ, [M*N,D]);
    ImageYCC = (M_TF * ImageXYZ')';
    ImageYCC = reshape(ImageYCC, [M,N,D]);
end


% YCC-RIT -> XYZ transformation (for images)
function ImageXYZ = ImageYCCRIT2XYZ(ImageYCC, M_TF)
    [M,N,D] = size(ImageYCC);
    ImageYCC = reshape(ImageYCC, [M*N,D]);
    ImageXYZ = (M_TF \ ImageYCC')';
    ImageXYZ = reshape(ImageXYZ, [M,N,D]);
end


% CSF for the luminance channel
% f = cycles per degree (cpd)
% s = sensitivity
%
% NOTE: Our luminance CSF is derived from that proposed by Reinhard et al.
% in "Color Imaging - Fundamentals and Applications". Unfortunately, this
% CSF leads to intensity shifts in the resulting image due to its bandpass
% nature. In addition, it goes above 1 for frequencies around 10 cpd, which
% causes an unnatural amplification of these frequencies in the resulting
% image.
% In our opinion, a CSF should attenuate frequencies below the visibility
% threshold, but it should not amplify frequencies our visual system is
% especially sensitive to.
% We solve this problem following a suggestion by Johnson and Fairchild
% and transform the bandpass filter into a lowpass filter. This is done by
% setting all y-values left to a certain x-value (the clipping threshold)
% to 1 (values below 1 as well as values above 1).
function s = CSF_lum(f)

% CSF parameters
a = 0.63;
b = 0.085;
c = 0.616;

% Clipping threshold
clipx = 13.3395;                % 2nd 1-crossing of the original function
                                % (as computed using Wolfram Alpha)

% CSF computation (lowpass filter)
s = a .* (f.^c) .* exp(-b.*f);  % f >  clipx = [Movshon and Kiorpes]
s(f <= clipx) = 1;            	% f <= clipx = 1 (no amplification)

% The luminance CSF does not need to be normalized; it is constant (= 1)
% within [0, clipx] and strictly monotonically decreasing afterwards.

end


% CSF for the red/green channel
% f = cycles per degree (cpd)
% s = sensitivity
function s = CSF_rg(f)

% CSF parameters
a1 = 91.228;
b1 = 0.0003;
c1 = 2.803;
a2 = 74.907;
b2 = 0.0038;
c2 = 2.601;

% CSF computation
s = a1 .* exp(-b1.*(f.^c1)) + a2 .* exp(-b2.*(f.^c2));

% Normalization: the chrominance CSFs need to be normalized so that
% CSF_chrom(0) = 1 to avoid DC amplification. For all other input values
% (f > 0) the function returns values < 1 so this doesn't pose a problem.
s = s./(a1+a2);

end


% CSF for the blue/yellow channel
% f = cycles per degree (cpd)
% s = sensitivity
function s = CSF_by(f)
        
% CSF parameters
a1 = 5.623;
b1 = 0.00001;
c1 = 3.4066;
a2 = 41.9363;
b2 = 0.083;
c2 = 1.3684;

% CSF computation
s = a1 .* exp(-b1.*(f.^c1)) + a2 .* exp(-b2.*(f.^c2));

% Normalization: the chrominance CSFs need to be normalized so that
% CSF_chrom(0) = 1 to avoid DC amplification. For all other input values
% (f > 0) the function returns values < 1 so this doesn't pose a problem.
s = s./(a1+a2);

end