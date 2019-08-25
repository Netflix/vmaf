clear all; close all; clc

inpath = 'E:\databaserelease2\databaserelease2\refimgs\';
fname = {'bikes.bmp'; 'buildings.bmp'; 'caps.bmp'; 'house.bmp'; 'lighthouse2.bmp'; 'monarch.bmp';
    'ocean.bmp'; 'paintedhouse.bmp'; 'parrots.bmp'; 'plane.bmp'; 'rapids.bmp'; 'sailing1.bmp'; 'sailing4.bmp';
    'stream.bmp'; 'building2.bmp'; 'cemetry.bmp'; 'churchandcapitol.bmp'; 'flowersonih35.bmp';
    'house.bmp'; 'lighthouse.bmp'; 'manfishing.bmp'};

% inpath = 'E:\images\';
% fname = {'barbara.png'; 'barco.png'; 'boats.png'; 'fingerprint.png'; 'flintstones.png';
%     'house.png'; 'lena.png'; 'peppers256.png'; };

s = size(fname);
blk=8;
path('D:\GSM\matlabPyrTools\',path);
Nsc = 4; Nor = 3;

Nband = Nsc*Nor+1;

mser1 = zeros(s(1),2); mser2 = zeros(s(1),2);
for u=5:5

    num=1; mse1 = zeros(Nband,2); mse2 = zeros(Nband,2);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Loading the image, splitting into halves and wavelet transform
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    pic_name = fname(u,1);
    fname1 = strcat(inpath,pic_name);
    temp1 = char(fname1);
    a = imread(temp1);
    a = rgb2gray(a);

    a1 = a(1:end,1:end/2);
    a2 = a(1:end,end:-1:end/2+1);
    %         a2 = a(1:end,end/2+1:end);

    %     figure; imshow(a1);
    %     figure; imshow(a2);
    a1 = double(a1);
    a2 = double(a2);

    [C1,S1] = buildSFpyr(a1,Nsc,Nor);
    [C2,S2] = buildSFpyr(a2,Nsc,Nor);
    Nband = size(S1,1);

    for nband = 1:Nband-1
        % Obtaining the desired wavelet coefficients
        wcoef1 = pyrBand(C1, S1, nband);
        wcoef2 = pyrBand(C2, S2, nband);

        s1 = size(wcoef1);
        modx = mod(s1(1),blk); mody = mod(s1(2),blk);
        coef1 = wcoef1; coef2 = wcoef2;
        wcoef1 = wcoef1(1:end-modx,1:end-mody); wcoef2 = wcoef2(1:end-modx,1:end-mody);
        s1 = size(wcoef1);
        count=1;

        % Estimating model paramaters
        [mu,covx,vars] = est_parameters(wcoef1,wcoef2,blk,s1);
        rho(nband) = covx(1,2)/sqrt(covx(1,1)*covx(2,2));
    end
end
rho

