% RES = STEER(BASIS, ANGLE, HARMONICS, STEERMTX)
%
% Steer BASIS to the specfied ANGLE.  
% 
% BASIS should be a matrix whose columns are vectorized rotated copies of a 
% steerable function, or the responses of a set of steerable filters.
% 
% ANGLE can be a scalar, or a column vector the size of the basis.
% 
% HARMONICS (optional, default is N even or odd low frequencies, as for 
% derivative filters) should be a list of harmonic numbers indicating
% the angular harmonic content of the basis.
% 
% STEERMTX (optional, default assumes cosine phase harmonic components,
% and filter positions at 2pi*n/N) should be a matrix which maps
% the filters onto Fourier series components (ordered [cos0 cos1 sin1 
% cos2 sin2 ... sinN]).  See steer2HarmMtx.m

% Eero Simoncelli, 7/96.

function res = steer(basis,angle,harmonics,steermtx)

num = size(basis,2);

if ( any(size(angle) ~= [size(basis,1) 1]) & any(size(angle) ~= [1 1]) )
  error('ANGLE must be a scalar, or a column vector the size of the basis elements');
end

%% If HARMONICS are not passed, assume derivatives.
if (exist('harmonics') ~= 1)
  if (mod(num,2) == 0)
    harmonics = [0:(num/2)-1]'*2 + 1;
  else
    harmonics = [0:(num-1)/2]'*2;
  end
else
  harmonics = harmonics(:);
  if ((2*size(harmonics,1)-any(harmonics == 0)) ~= num)
    error('harmonics list is incompatible with basis size');
  end
end

%% If STEERMTX not passed, assume evenly distributed cosine-phase filters:
if (exist('steermtx') ~= 1)
  steermtx = steer2HarmMtx(harmonics, pi*[0:num-1]/num, 'even');
end

steervect = zeros(size(angle,1),num);
arg = angle * harmonics(find(harmonics~=0))';
if (all(harmonics))
	steervect(:, 1:2:num) = cos(arg);
	steervect(:, 2:2:num) = sin(arg);
else
	steervect(:, 1) = ones(size(arg,1),1);
	steervect(:, 2:2:num) = cos(arg);
	steervect(:, 3:2:num) = sin(arg);
end

steervect = steervect * steermtx;

if (size(steervect,1) > 1)
	tmp = basis' .* steervect';
	res = sum(tmp)';
else
	res = basis * steervect';
end
	
	
