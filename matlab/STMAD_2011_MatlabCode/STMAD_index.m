function MadVals = STMAD_index(OrgFile, DstFile, Wid, Hei)
%-----------------------------------------------------------------------
% Function to compute the Spatiotemporal Most Apparent Distortion STMAD index. 
% The input files must be YUV420P with dimension WidxHei.
% OrgFile: The original (reference) video file.
% DstFile: The distorted video file.
% MadVals: The output value including SMAD, TMAD and STMAD.

% Author: Phong Vu
% Computational Perception and Image Quality Lab
% Department of Electrical and Computer Engineering
% Oklahoma State University

% 2011: First released version. The new version will be updated soon.

% Please cite the following papers if you wish to use our code:

% 1. E.C. Larson and D.M. Chandler, “Most apparent distortion: Full-reference image quality assessment 
% and the role of strategy,” Journal of Electronic Imaging, vol. 19, no. 1, 2010.
% 
% 2. P.V. Vu, C.T. Vu and D.M. Chandler, "A Spatiotemporal Most Apparent Distortion Model for Video
% Quality Assessment", 18th IEEE International Conference on Image Processing (ICIP), Sep 2011.

%-----------------------------------------------------------------------
% Reading video files

disp(OrgFile);
disp(DstFile);

OrgYuv = yuvread(OrgFile, Wid, Hei, '420');
DstYuv = yuvread(DstFile, Wid, Hei, '420');

len = length(DstYuv);

% Compute Spatial MAD
HiIndex = zeros(len, 1);
LoIndex = zeros(len, 1);

fprintf('\n 1. Computing SMAD: \n');

for idx = 1:len
    fprintf(' %3d', idx);
    if rem(idx,20) == 0, fprintf(' \n');
    end
    OrgImg = double(OrgYuv(idx).cdata(:,:,1));
    DstImg = double(DstYuv(idx).cdata(:,:,1));
    
    HiIndex(idx) = hi_index(OrgImg, DstImg);
    LoIndex(idx) = lo_index(OrgImg, DstImg);    
end

MadHi = mean(HiIndex);
MadLo = mean(LoIndex);

b1        = exp(-2.55/3.35);
b2        = 1/(log(10)*3.35);
alpha     = 3/7;
       
sig       = 1 ./ ( 1 + b1*(MadHi).^b2 ) ;

MadVals.SMAD = MadHi.^(sig/4) + MadLo.^(1-sig);

% Compute Motion Weights using optical flow method

M = MotionWeight(OrgYuv, Wid, Hei);

tmpCol = M.wCol;
mCol = sum(tmpCol);
if mCol == 0, 
    mCol = 1;
    tmpCol = 1/length(tmpCol)* ones(size(tempCol));
end
    
tmpRow = M.wRow;
mRow = sum(tmpRow);
if mRow == 0, 
    mRow = 1;
    tmpRow = 1/length(tmpRow)* ones(size(tempRow));
end

% Compute Temporal MAD for 1 of every 8 STS images

nrows = floor(Hei/8);
ncols = floor(Wid/8);
mad_rows = zeros(nrows,1);
mad_cols = zeros(ncols,1);

fprintf('\n 2. Computing TMAD:');
% Extract STS column images and compute d_appear index
Org_STS_Col = zeros(Hei,len);
Dst_STS_Col = zeros(Hei,len);

fprintf('\n a. STS column: \n');

for t1=1:ncols
    colid = 8*t1-3;
    fprintf(' %3d', colid);
    if rem(t1,20) == 0, fprintf(' \n');
    end
    for idx = 1:len
        Org_STS_Col(:,idx) = uint8(OrgYuv(idx).cdata(:,colid,1));
        Dst_STS_Col(:,idx) = uint8(DstYuv(idx).cdata(:,colid,1));
    end
          
    mad_cols(t1) = lo_index(Org_STS_Col, Dst_STS_Col);
end

% Extract STS row images and compute d_appear index
Org_STS_Row = zeros(len, Wid);
Dst_STS_Row = zeros(len, Wid);

fprintf('\n b. STS row: \n');

for t2=1:nrows
    rowid = 8*t2-3;
    fprintf(' %3d', rowid);
    if rem(t2,20) == 0, fprintf(' \n');
    end
    for idx = 1:len
        Org_STS_Row(idx,:) = uint8(OrgYuv(idx).cdata(rowid,:,1));
        Dst_STS_Row(idx,:) = uint8(DstYuv(idx).cdata(rowid,:,1));
    end
    
    mad_rows(t2) = lo_index(Org_STS_Row, Dst_STS_Row);
end

fprintf(' \n');

stsCol = log10(1000*alpha + mCol) * sum(tmpCol.*mad_cols)/mCol;
stsRow = log10(1000*alpha + mRow) * sum(tmpRow.*mad_rows)/mRow;
       
MadVals.TMAD = stsRow^alpha + stsCol^(1-alpha);

% Compute ST-MAD index

movIndex = mRow / (mRow + mCol);
beta  = log10(1+movIndex);
MadVals.STMAD = 2.5 * log10(beta.*MadVals.SMAD) + MadVals.TMAD;

