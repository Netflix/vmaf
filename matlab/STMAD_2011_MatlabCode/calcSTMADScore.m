function calcSTMADScore(OrgFile,DstFile,Wid,Hei)

OrgYuv = yuvread(OrgFile, Wid, Hei, '420');
DstYuv = yuvread(DstFile, Wid, Hei, '420');

len = length(DstYuv);

% Compute Spatial MAD
HiIndex = zeros(len, 1);
LoIndex = zeros(len, 1);

for idx = 1:len

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

% Extract STS column images and compute d_appear index
Org_STS_Col = zeros(Hei,len);
Dst_STS_Col = zeros(Hei,len);

for t1=1:ncols
    
    colid = 8*t1-3;

    for idx = 1:len
        Org_STS_Col(:,idx) = uint8(OrgYuv(idx).cdata(:,colid,1));
        Dst_STS_Col(:,idx) = uint8(DstYuv(idx).cdata(:,colid,1));
    end
          
    mad_cols(t1) = lo_index(Org_STS_Col, Dst_STS_Col);
end

% Extract STS row images and compute d_appear index
Org_STS_Row = zeros(len, Wid);
Dst_STS_Row = zeros(len, Wid);

for t2=1:nrows
    
    rowid = 8*t2-3;
    for idx = 1:len
        Org_STS_Row(idx,:) = uint8(OrgYuv(idx).cdata(rowid,:,1));
        Dst_STS_Row(idx,:) = uint8(DstYuv(idx).cdata(rowid,:,1));
    end
    
    mad_rows(t2) = lo_index(Org_STS_Row, Dst_STS_Row);
    
end

stsCol = log10(1000*alpha + mCol) * sum(tmpCol.*mad_cols)/mCol;
stsRow = log10(1000*alpha + mRow) * sum(tmpRow.*mad_rows)/mRow;
       
MadVals.TMAD = stsRow^alpha + stsCol^(1-alpha);

% Compute ST-MAD index

movIndex = mRow / (mRow + mCol);
beta  = log10(1+movIndex);
MadVals.STMAD = 2.5 * log10(beta.*MadVals.SMAD) + MadVals.TMAD;

for iframe = 0 : len-1
    disp(sprintf('smad: %d %f', iframe, MadVals.SMAD));
    disp(sprintf('tmad: %d %f', iframe, MadVals.TMAD));
    disp(sprintf('stmad: %d %f', iframe, MadVals.STMAD));
end;



   
