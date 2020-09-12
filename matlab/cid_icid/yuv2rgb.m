function rgb=yuv2rgb(Y,U,V,yuvformat,convmtrx)
%Converts YUV to RGB
%rgb=yuv2rgb(Y,U,V,yuvformat,convmtrx)
%
%Input:
% Y,U,V - Y,U and V components of the frame
% yuvformat - YUV format [optional, default = 'YUV420_8']. See in rgb2yuv.m
%             for supported YUV subsampling formats. 
% convmtrx - Conversion matrix [optional, default = 'BT709_l']. The 
%            following conversions ase defined. See in rgb2yuv.m
%             for more details.
%
%Output:
% rgb - RGB 3D matrix. rgb(:,:,1), rgb(:,:,2) and rgb(:,:,3) are R, G and
%       B components, respectively.
%
%Uses:
% imresize.m - Matlab Image Processing Toolbox
%
%Note:
% When the input format has the chroma subsampled (e.g. 4:2:0 format), 
% upsampling is employed on the chroma components before the conversion 
% takes place. 
% See in the help of rgb2yuv.m for details on coversion options.
%
% ITU-R BT.601, RGB full range, results in the following transform matrix:
%  1.164   0.000   1.596
%  1.164  -0.392  -0.813
%  1.164   2.017   0.000
% ITU-R BT.601, RGB limited range, results in the following transform matrix:
%  1.000   0.000   1.402
%  1.000  -0.344  -0.714
%  1.000   1.772   0.000
% ITU-R BT.709, RGB limited range, results in the following transform matrix:
%  1.000   0.000   1.570
%  1.000  -0.187  -0.467
%  1.000   1.856   0.000
%  
%Example:
% rgb = yuv2rgb(Y,U,V);

if (nargin < 4)
    yuvformat = 'YUV420_8';
end;
if (nargin < 5)
    convmtrx = 'BT709_l';
end;
if (strcmp(yuvformat,'YUV420_8') && (exist('imresize','file') ~= 2))
    error('For YUV420 subsampling yuv2rgb requires Image Processing Toolbox (TM) function imresize!');
end;

if strcmp(convmtrx,'BT601_f')
   load('BT601_f.mat','-mat');
elseif strcmp(convmtrx,'BT601_l')
   load('BT601_l.mat','-mat');
elseif strcmp(convmtrx,'BT601_219')
   load('BT601_219.mat','-mat');
elseif strcmp(convmtrx,'BT709_f')
   load('BT709_f.mat','-mat');
elseif strcmp(convmtrx,'BT709_l')
   load('BT709_l.mat','-mat');
end;

%create the 3D YUV array
yuv = zeros(size(Y,1),size(Y,2),3);
if (strcmp(yuvformat,'YUV420_8'))
    yuv(:,:,1) = double(Y);
    yuv(:,:,2) = imresize(double(U),2,'bicubic');
    yuv(:,:,3) = imresize(double(V),2,'bicubic');
elseif (strcmp(yuvformat,'YUV444_8'))
    yuv(:,:,1) = double(Y);
    yuv(:,:,2) = double(U);
    yuv(:,:,3) = double(V);
end;

%inversion of the transform matrix
T = inv(rgb2yuvT);
rgb = zeros(size(Y,1),size(Y,2),3);
if (yuvoffset(1) ~= 0)
  yuv(:,:,1) = yuv(:,:,1) - yuvoffset(1);
end;
if (yuvoffset(2) ~= 0)
  yuv(:,:,2) = yuv(:,:,2) - yuvoffset(2);
end;
if (yuvoffset(3) ~= 0)
  yuv(:,:,3) = yuv(:,:,3) - yuvoffset(3);
end;
rgb(:,:,1) = T(1,1) * yuv(:,:,1) + T(1,2) * yuv(:,:,2) + T(1,3) * yuv(:,:,3);
rgb(:,:,2) = T(2,1) * yuv(:,:,1) + T(2,2) * yuv(:,:,2) + T(2,3) * yuv(:,:,3);
rgb(:,:,3) = T(3,1) * yuv(:,:,1) + T(3,2) * yuv(:,:,2) + T(3,3) * yuv(:,:,3);
rgb = uint8(round(rgb));