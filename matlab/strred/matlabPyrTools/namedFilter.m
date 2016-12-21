% KERNEL = NAMED_FILTER(NAME)
%
% Some standard 1D filter kernels.  These are scaled such that
% their L2-norm is 1.0.
%
%  binomN             - binomial coefficient filter of order N-1
%  haar:              - Haar wavelet.
%  qmf8, qmf12, qmf16 - Symmetric Quadrature Mirror Filters [Johnston80]
%  daub2,daub3,daub4  - Daubechies wavelet [Daubechies88].
%  qmf5, qmf9, qmf13: - Symmetric Quadrature Mirror Filters [Simoncelli88,Simoncelli90]
%
%  See bottom of file for full citations.

% Eero Simoncelli, 6/96.

function [kernel] = named_filter(name)

if strcmp(name(1:min(5,size(name,2))), 'binom')
  kernel = sqrt(2) * binomialFilter(str2num(name(6:size(name,2))));
elseif strcmp(name,'qmf5')
  kernel = [-0.076103 0.3535534 0.8593118 0.3535534 -0.076103]';
elseif strcmp(name,'qmf9')
  kernel = [0.02807382 -0.060944743 -0.073386624 0.41472545 0.7973934 ...
      0.41472545 -0.073386624 -0.060944743 0.02807382]';
elseif strcmp(name,'qmf13')
  kernel = [-0.014556438 0.021651438 0.039045125 -0.09800052 ...
	  -0.057827797 0.42995453 0.7737113 0.42995453 -0.057827797 ...
	  -0.09800052 0.039045125 0.021651438 -0.014556438]';
elseif strcmp(name,'qmf8')
  kernel = sqrt(2) * [0.00938715 -0.07065183 0.06942827 0.4899808 ...
    0.4899808 0.06942827 -0.07065183 0.00938715 ]';
elseif strcmp(name,'qmf12')
  kernel = sqrt(2) * [-0.003809699 0.01885659 -0.002710326 -0.08469594 ...
	0.08846992 0.4843894 0.4843894 0.08846992 -0.08469594 -0.002710326 ...
	0.01885659 -0.003809699 ]';
elseif strcmp(name,'qmf16')
  kernel = sqrt(2) * [0.001050167 -0.005054526 -0.002589756 0.0276414 -0.009666376 ...
	-0.09039223 0.09779817 0.4810284 0.4810284 0.09779817 -0.09039223 -0.009666376 ...
	0.0276414 -0.002589756 -0.005054526 0.001050167 ]';
elseif strcmp(name,'haar')
  kernel = [1 1]' / sqrt(2);
elseif strcmp(name,'daub2')
  kernel = [0.482962913145 0.836516303738 0.224143868042 -0.129409522551]';
elseif strcmp(name,'daub3')
  kernel = [0.332670552950 0.806891509311 0.459877502118 -0.135011020010 ...
	-0.085441273882  0.035226291882]';
elseif strcmp(name,'daub4')
  kernel = [0.230377813309 0.714846570553 0.630880767930 -0.027983769417 ...
	-0.187034811719 0.030841381836 0.032883011667 -0.010597401785]';
elseif strcmp(name,'gauss5')  % for backward-compatibility
  kernel = sqrt(2) * [0.0625 0.25 0.375 0.25 0.0625]';
elseif strcmp(name,'gauss3')  % for backward-compatibility
  kernel = sqrt(2) * [0.25 0.5 0.25]';
else
  error(sprintf('Bad filter name: %s\n',name));
end
  
% [Johnston80] - J D Johnston, "A filter family designed for use in quadrature 
%    mirror filter banks", Proc. ICASSP, pp 291-294, 1980.
%
% [Daubechies88] - I Daubechies, "Orthonormal bases of compactly supported wavelets",
%    Commun. Pure Appl. Math, vol. 42, pp 909-996, 1988.
%
% [Simoncelli88] - E P Simoncelli,  "Orthogonal sub-band image transforms",
%     PhD Thesis, MIT Dept. of Elec. Eng. and Comp. Sci. May 1988.
%     Also available as: MIT Media Laboratory Vision and Modeling Technical 
%     Report #100.
%
% [Simoncelli90] -  E P Simoncelli and E H Adelson, "Subband image coding",
%    Subband Transforms, chapter 4, ed. John W Woods, Kluwer Academic 
%    Publishers,  Norwell, MA, 1990, pp 143--192.
