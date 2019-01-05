function [I mp imgs] = hi_index( ref_img, dst_img )
% Calculate the high quality index by calculating the masking map and
% then approximating the local statistics, using filtering.
% Author: Eric Larson
% Department of Electrical and Computer Engineering
% Oklahoma State University, 2008
% Image Coding and Analysis Lab

% Masking/luminance parameters
k = 0.02874;
G = 0.5 ;       % luminance threshold
C_slope = 1;    % slope of detection threshold
Ci_thrsh= -5;   % contrast to start slope, rather than const threshold
Cd_thrsh= -5;   % saturated threshold
ms_scale= 1;    % scaling constant
if(nargin==0)% for debug only,
  I={'collapsing (mask) and raw lmse',...
    'using c code',...
    'using log scale',...
    'two norm',...
    sprintf('Ci = %.2f',Ci_thrsh),...
    sprintf('Cslope = %.2f',C_slope),...
    sprintf('Cd = %.2f',Cd_thrsh),...
    sprintf('G = %.2f',G),...
    'LO QUALITY',...
    };
  return;
end
% log(Contrast of  ref-dst)   vs.   log(Contrast of reference)
%              /
%             /
%            /  _| <- Cslope
%           /
%----------+ <- Cdthresh (y axis height)
%          /\
%          ||
%       Ci thresh (x axis value)

% TAKE TO LUMINANCE DOMAIN USING LUT
if isinteger(ref_img)
  LUT = 0:1:255; %generate LUT
  LUT = k .* LUT .^ (2.2/3);
  ref = LUT( ref_img + 1 );
  dst = LUT( dst_img + 1 );
else % don't use the speed up
  ref = k .* ref_img .^ (2.2/3);
  dst = k .* dst_img .^ (2.2/3);
end

[M N] = size( ref );

% ACCOUNT FOR CONTRAST SENSITIVITY
csf = make_csf( M, N, 32 )';
ref = real( ifft2( ifftshift( fftshift( fft2( ref ) ).* csf ) ) );
dst = real( ifft2( ifftshift( fftshift( fft2( dst ) ).* csf ) ) );
refS = ref;
dstS = dst;

% Use c code to get fast local stats
[std_2 std_1 m1_1] = ical_std( dst-ref, ref );

BSIZE = 16;

Ci_ref = log(std_1./m1_1); % contrast of reference (also a measure of entropy)
Ci_dst = log(std_2./m1_1); % contrast of distortions (also a measure of entropy)
Ci_dst( find( m1_1 < G ) ) = -inf;

msk       = zeros( size(Ci_dst) );
idx1      = find( (Ci_ref > Ci_thrsh) ...
  & (Ci_dst > (C_slope * (Ci_ref - Ci_thrsh) + Cd_thrsh) ) );
idx2      = find( (Ci_ref <= Ci_thrsh) & (Ci_dst > Cd_thrsh) );

msk(idx1) = ( Ci_dst( idx1 ) - (C_slope * (Ci_ref( idx1 )-Ci_thrsh) + Cd_thrsh) ) ./ ms_scale;
msk(idx2) = ( Ci_dst( idx2 ) - Cd_thrsh ) ./ ms_scale;
%= ( Contrast of heavy Dst - 0.75 * Contrast Ref ) / normalize
%= ( Contrast of low Dst  - Threshold ) / normalize

% Use lmse and weight by distortion mask
win = ones( BSIZE ) ./ BSIZE^2;
lmse  = ( imfilter( ( double(ref_img) - double(dst_img) ).^2, ...
  win,  'symmetric', 'same', 'conv' ) );

mp    = msk .* lmse;

% kill the edges
mp2 = mp( BSIZE+1:end-BSIZE-1, BSIZE+1:end-BSIZE-1);

I = norm( mp2(:) , 2 ) ./ sqrt( length( mp2(:) ) ) .* 10;

if( nargout > 2)
  imgs.ref = refS;
  imgs.dst = dstS;
end


end
%-----------------------------------------------------

function [res] = make_csf(x, y, nfreq)
[xplane,yplane]=meshgrid(-x/2+0.5:x/2-0.5, -y/2+0.5:y/2-0.5);	% generate mesh
plane=(xplane+1i*yplane)/y*2*nfreq;
radfreq=abs(plane);				% radial frequency

% We modify the radial frequency according to angle.
% w is a symmetry parameter that gives approx. 3 dB down along the
% diagonals.
w=0.7;
s=(1-w)/2*cos(4*angle(plane))+(1+w)/2;
radfreq=radfreq./s;

% Now generate the CSF
csf = 2.6*(0.0192+0.114*radfreq).*exp(-(0.114*radfreq).^1.1);
f=find( radfreq < 7.8909 ); csf(f)=0.9809+zeros(size(f));

res = csf;
end
%-----------------------------------------------------
