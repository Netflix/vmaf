%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Some examples using the tools in this distribution.
%%% Eero Simoncelli, 2/97.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Add directory to path (YOU'LL NEED TO ADJUST THIS):
path('/lcv/matlab/lib/matlabPyrTools',path);

%% Load an image, and downsample to a size appropriate for the machine speed.
oim = pgmRead('einstein.pgm');
tic; corrDn(oim,[1 1; 1 1]/4,'reflect1',[2 2]); time = toc;
imSubSample = min(max(floor(log2(time)/2+3),0),2);
im = blurDn(oim, imSubSample,'qmf9');
clear oim;

%%% ShowIm: 
%% 3 types of automatic graylevel scaling, 2 types of automatic
%% sizing, with or without title and Range information.
help showIm
clf; showIm(im,'auto1','auto','Al')
clf; showIm('im','auto2')
clf; showIm(im,'auto3',2)

%%% Statistics:
mean2(im)
var2(im)
skew2(im)
kurt2(im)
entropy2(im)
imStats(im)

%%% Synthetic images.  First pick some parameters:
sz = 200;
dir = 2*pi*rand(1)
slope = 10*rand(1)-5
int = 10*rand(1)-5;
orig = round(1+(sz-1)*rand(2,1));
expt = 0.8+rand(1)
ampl = 1+5*rand(1)
ph = 2*pi*rand(1)
per = 20
twidth = 7

clf;
showIm(mkRamp(sz,dir,slope,int,orig));
showIm(mkImpulse(sz,orig,ampl));
showIm(mkR(sz,expt,orig));
showIm(mkAngle(sz,dir));
showIm(mkDisc(sz,sz/4,orig,twidth));
showIm(mkGaussian(sz,(sz/6)^2,orig,ampl));
showIm(mkZonePlate(sz,ampl,ph));
showIm(mkAngularSine(sz,3,ampl,ph,orig));
showIm(mkSine(sz,per,dir,ampl,ph,orig));
showIm(mkSquare(sz,per,dir,ampl,ph,orig,twidth));
showIm(mkFract(sz,expt));


%%% Point operations (lookup tables):
[Xtbl,Ytbl] = rcosFn(20, 25, [-1 1]);
plot(Xtbl,Ytbl);
showIm(pointOp(mkR(100,1,[70,30]), Ytbl, Xtbl(1), Xtbl(2)-Xtbl(1), 0));


%%% histogram Modification/matching:
[N,X] = histo(im, 150);
[mn, mx] = range2(im);
matched = histoMatch(rand(size(im)), N, X);
showIm(im + sqrt(-1)*matched);
[Nm,Xm] = histo(matched,150);
nextFig(2,1); 
  subplot(1,2,1); plot(X,N); axis([mn mx 0 max(N)]);
  subplot(1,2,2);  plot(Xm,Nm); axis([mn mx 0 max(N)]);
nextFig(2,-1);

%%% Convolution routines:

%% Compare speed of convolution/downsampling routines:
noise = rand(400); filt = rand(10);
tic; res1 = corrDn(noise,filt(10:-1:1,10:-1:1),'reflect1',[2 2]); toc;
tic; ires = rconv2(noise,filt); res2 = ires(1:2:400,1:2:400); toc;
imStats(res1,res2)

%% Display image and extension of left and top boundaries:
fsz = [9 9];
fmid = ceil((fsz+1)/2);
imsz = [16 16];

% pick one:
im = eye(imsz);
im = mkRamp(imsz,pi/6); 
im = mkSquare(imsz,6,pi/6); 

% pick one:
edges='reflect1';
edges='reflect2';
edges='repeat';
edges='extend';
edges='zero';
edges='circular';
edges='dont-compute';

filt = mkImpulse(fsz,[1 1]);
showIm(corrDn(im,filt,edges));
line([0,0,imsz(2),imsz(2),0]+fmid(2)-0.5, ...
     [0,imsz(1),imsz(1),0,0]+fmid(1)-0.5);
title(sprintf('Edges = %s',edges));

%%% Multi-scale pyramids (see pyramids.m for more examples,
%%% and explanations):

%% A Laplacian pyramid:
[pyr,pind] = buildLpyr(im);
showLpyr(pyr,pind);

res = reconLpyr(pyr, pind); 		% full reconstruction
imStats(im,res);			% essentially perfect

res = reconLpyr(pyr, pind, [2 3]);  %reconstruct 2nd and 3rd levels only  
showIm(res);

%% Wavelet/QMF pyramids:
filt = 'qmf9'; edges = 'reflect1';
filt = 'haar'; edges = 'qreflect2';
filt = 'qmf12'; edges = 'qreflect2';
filt = 'daub3'; edges = 'circular';

[pyr,pind] = buildWpyr(im, 5-imSubSample, filt, edges);
showWpyr(pyr,pind,'auto2');

res = reconWpyr(pyr, pind, filt, edges);
clf; showIm(im + i*res);
imStats(im,res);

res = reconWpyr(pyr, pind, filt, edges, 'all', [2]);  %vertical only
clf; showIm(res);

%% Steerable pyramid:
[pyr,pind] = buildSpyr(im,4-imSubSample,'sp3Filters');  
showSpyr(pyr,pind);

%% Steerable pyramid, constructed in frequency domain:
[pyr,pind] = buildSFpyr(im,5-imSubSample,4);  %5 orientation bands
showSpyr(pyr,pind);
res = reconSFpyr(pyr,pind);
imStats(im,res);
