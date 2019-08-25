% IM = mkSine(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN)
%      or
% IM = mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN)
% 
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a 2D sinusoid, with given PERIOD (in pixels),
% DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE (default
% = 1), and PHASE (radians, relative to ORIGIN, default = 0).  ORIGIN
% defaults to the center of the image.
% 
% In the second form, FREQ is a 2-vector of frequencies (radians/pixel).

% Eero Simoncelli, 6/96.

function [res] = mkSine(sz, per_freq, dir_amp, amp_phase, phase_orig, orig)

%------------------------------------------------------------
%% OPTIONAL ARGS:

if (prod(size(per_freq)) == 2)
  frequency = norm(per_freq);
  direction = atan2(per_freq(1),per_freq(2));
  if (exist('dir_amp') == 1)
    amplitude = dir_amp;
  else
    amplitude = 1;
  end
  if (exist('amp_phase') == 1)
    phase = amp_phase;
  else
    phase = 0;
  end
  if (exist('phase_orig') == 1)
    origin = phase_orig;
  end
  if (exist('orig') == 1)
    error('Too many arguments for (second form) of mkSine');
  end
else
  frequency = 2*pi/per_freq;
  if (exist('dir_amp') == 1)
    direction = dir_amp;
  else
    direction = 0;
  end
  if (exist('amp_phase') == 1)
    amplitude = amp_phase;
  else
    amplitude = 1;
  end
  if (exist('phase_orig') == 1)
    phase = phase_orig;
  else
    phase = 0;
  end
  if (exist('orig') == 1)
    origin = orig;
  end
end

%------------------------------------------------------------
 
if (exist('origin') == 1)
  res = amplitude*sin(mkRamp(sz, direction, frequency, phase, origin));
else
 res = amplitude*sin(mkRamp(sz, direction, frequency, phase));
end
