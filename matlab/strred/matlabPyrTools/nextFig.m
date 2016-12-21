% nextFig (MAXFIGS, SKIP)
% 
% Make figure number mod((GCF+SKIP), MAXFIGS) the current figure.
% MAXFIGS is optional, and defaults to 2.
% SKIP is optional, and defaults to 1.

% Eero Simoncelli, 2/97.

function nextFig(maxfigs, skip)

if (exist('maxfigs') ~= 1)
  maxfigs = 2;
end
  
if (exist('skip') ~= 1)
  skip = 1;
end
  
figure(1+mod(gcf-1+skip,maxfigs));
