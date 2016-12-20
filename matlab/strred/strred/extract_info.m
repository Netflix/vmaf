function [spatial temporal] = extract_info(frame1,frame2)

% Copyright (c) 2011 The University of Texas at Austin
% All rights reserved.
%
% Permission to use, copy, or modify this software and its documentation for
% educational and research purposes only and without fee is hereby granted,
% provided that this copyright notice and the original authors' names appear on
% all copies and supporting documentation. This program shall not be used, rewritten,
% or adapted as the basis of a commercial software or hardware product without first
% obtaining permission of the authors. The authors make no representations about the
% suitability of this software for any purpose. It is provided "as is" without express
% or implied warranty.

% The following paper is to be cited in the bibliography whenever the software is used
% as:
% R. Soundararajan and A. C. Bovik, "RRED indices: Reduced reference entropic
% differences for image quality assessment", IEEE Transactions on Image
% Processing, vol. 21, no. 2, pp. 517-526, Feb. 2012

% The code used herein is developed using code already developed for the
% algorithms in the following two papers:
% H. R. Sheikh and A. C. Bovik, "Image information and visual quality," IEEE Trans. Image Process., vol. 15, no. 2, Feb. 2006
% Q. Li and Z. Wang, "Reduced-reference image quality assessment using divisive normalization-based image representation," IEEE Journal of Selected Topics in Signal Processing: Special issue on Visual Media Quality Assessment, vol. 3, 2009.

% 'spatial' and 'temporal' refer to the spatial and temporal scaled entropy information
% for different blocks in the subband

blk=3; %Size of block
sigma_nsq = 0.1; sigma_nsqt = 0.1; %Neural noise variances
band=4; %Vertically oriented subband in the coarsest scale

path('matlabPyrTools/',path);

%Wavelet decompositions using steerable pyramids
[pyr,pind] = buildSpyr(double(frame1), 4, 'sp5Filters', 'reflect1');
dframe=ind2wtree(pyr,pind);
clear pyr
y1 = dframe{band};
[pyr,pind] = buildSpyr(double(frame2), 4, 'sp5Filters', 'reflect1');
dframe=ind2wtree(pyr,pind);
clear pyr
y2 = dframe{band};
ydiff = y1-y2;

%estimate the entropy at different locations and the local spatial/temporal
%premultipliers
[ss q] = est_params(y1,blk,sigma_nsq);%Spatial
[ssdiff qdiff] = est_params(ydiff,blk,sigma_nsqt); %Temporal

spatial = q.*log2(1+ss);

temporal = qdiff.*log2(1+ss).*log2(1+ssdiff);


