% IM = mkFract(SIZE, FRACT_DIM)
%
% Make a matrix of dimensions SIZE (a [Y X] 2-vector, or a scalar)
% containing fractal (pink) noise with power spectral density of the
% form: 1/f^(5-2*FRACT_DIM).  Image variance is normalized to 1.0.
% FRACT_DIM defaults to 1.0

% Eero Simoncelli, 6/96.

%% TODO: Verify that this  matches Mandelbrot defn of fractal dimension.
%%       Make this more efficient!

function res = mkFract(dims, fract_dim)

if (exist('fract_dim') ~= 1)
  fract_dim = 1.0;
end

res = randn(dims);
fres = fft2(res);

sz = size(res);
ctr = ceil((sz+1)./2);

shape = ifftshift(mkR(sz, -(2.5-fract_dim), ctr));
shape(1,1) = 1;  %%DC term

fres = shape .* fres;
fres = ifft2(fres);

if (max(max(abs(imag(fres)))) > 1e-10)
  error('Symmetry error in creating fractal');
else
  res = real(fres);
  res = res / sqrt(var2(res));
end  
