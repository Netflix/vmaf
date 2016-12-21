% IM = mkSquare(SIZE, PERIOD, DIRECTION, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
%      or
% IM = mkSine(SIZE, FREQ, AMPLITUDE, PHASE, ORIGIN, TWIDTH)
% 
% Compute a matrix of dimension SIZE (a [Y X] 2-vector, or a scalar)
% containing samples of a 2D square wave, with given PERIOD (in
% pixels), DIRECTION (radians, CW from X-axis, default = 0), AMPLITUDE
% (default = 1), and PHASE (radians, relative to ORIGIN, default = 0).
% ORIGIN defaults to the center of the image.  TWIDTH specifies width
% of raised-cosine edges on the bars of the grating (default =
% min(2,period/3)).
% 
% In the second form, FREQ is a 2-vector of frequencies (radians/pixel).

% Eero Simoncelli, 6/96.

% TODO: Add duty cycle.  

function [res] = mkSquare(sz, per_freq, dir_amp, amp_phase, phase_orig, orig_twidth, twidth)

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
  if (exist('orig_twidth') == 1)
    transition = orig_twidth;
  else
    transition = min(2,2*pi/(3*frequency));
  end
  if (exist('twidth') == 1)
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
  if (exist('orig_twidth') == 1)
    origin = orig_twidth;
  end
  if (exist('twidth') == 1)
    transition = twidth;
  else
    transition = min(2,2*pi/(3*frequency));
  end

end

%------------------------------------------------------------
  
if (exist('origin') == 1)
  res = mkRamp(sz, direction, frequency, phase, origin) - pi/2;
else
  res = mkRamp(sz, direction, frequency, phase) - pi/2;
end

[Xtbl,Ytbl] = rcosFn(transition*frequency,pi/2,[-amplitude amplitude]);

res = pointOp(abs(mod(res+pi, 2*pi)-pi),Ytbl,Xtbl(1),Xtbl(2)-Xtbl(1),0);

% OLD threshold version: 
%res = amplitude * (mod(res,2*pi) < pi);
