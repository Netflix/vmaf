function M = MotionWeight(yuvobj, Wid, Hei)

wSize = 8;

N=length(yuvobj);
r0=floor(Hei/wSize);
c0=floor(Wid/wSize);

tmp=zeros(r0,c0);

for idx=2:N

    yuv_img=yuvobj(idx).cdata(1:r0*wSize,1:c0*wSize,1);
    pre_img=yuvobj(idx-1).cdata(1:r0*wSize,1:c0*wSize,1);
    
    VH = LK_index( yuv_img, pre_img, wSize);
    tmp=tmp+VH;
    
end

tmp=tmp/(N-1);

mVer=mean(tmp,2);
mHor=mean(tmp,1)';

gVer = gausswin(r0, 1.2);
gHor = gausswin(c0, 1.2);

mRow = (mVer.^2).*gVer/sum(gVer);
mCol = (mHor.^2).*gHor/sum(gHor);

M.wRow = mRow;
M.wCol = mCol;


%--------------------------------------------------------------------------
function VH = LK_index(imgCurr, imgPrev, windowSize)

if (size(imgCurr,1) ~= size(imgPrev,1)) || (size(imgCurr,2) ~= size(imgPrev,2))
    error('input images are not the same size');
end;

if (size(imgCurr,3)~=1) || (size(imgPrev,3)~=1)
    error('method only works for gray-level images');
end;

imgCurr=double(imgCurr);
imgPrev=double(imgPrev);

fx = conv2(imgCurr,0.25* [-1 1; -1 1]) + conv2(imgPrev, 0.25*[-1 1; -1 1]);
fy = conv2(imgCurr, 0.25*[-1 -1; 1 1]) + conv2(imgPrev, 0.25*[-1 -1; 1 1]);
ft = conv2(imgCurr, 0.25*ones(2)) + conv2(imgPrev, -0.25*ones(2));

% make same size as input
fx=fx(1:size(fx,1)-1, 1:size(fx,2)-1);
fy=fy(1:size(fy,1)-1, 1:size(fy,2)-1);
ft=ft(1:size(ft,1)-1, 1:size(ft,2)-1);

% 8;
r0=floor(size(fx,1)/windowSize);
c0=floor(size(fx,2)/windowSize);

u = zeros(r0,c0);
v = zeros(r0,c0);

for i = 1:r0
    for j = 1:c0
        
        curFx = fx((i-1)*windowSize+1:i*windowSize, (j-1)*windowSize+1:j*windowSize);
        curFy = fy((i-1)*windowSize+1:i*windowSize, (j-1)*windowSize+1:j*windowSize);
        curFt = ft((i-1)*windowSize+1:i*windowSize, (j-1)*windowSize+1:j*windowSize);
        
        curFx = curFx';
        curFy = curFy';
        curFt = curFt';
        
        curFx = curFx(:);
        curFy = curFy(:);
        curFt = -curFt(:);
        
        A = [curFx curFy];
        U = pinv(A'*A)*A'*curFt;
        
        u(i,j)=U(1);
        v(i,j)=U(2);
        
    end;
end;

u(isnan(u))=0;
v(isnan(v))=0;

VH=u.^2+v.^2;