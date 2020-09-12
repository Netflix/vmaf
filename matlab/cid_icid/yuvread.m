
%	Filename --> Name of File (e.g. 'Test.yuv')
%   width    --> width of a frame  (e.g. 352)
%   height   --> height of a frame (e.g. 280)
%   format   --> subsampling rate ('400','411','420','422' or '444')
%
%example: yuv = yuvread('Test.yuv',352,288,'420');

function mov = yuvread(File,width,height,format, subsample)

    %set factor for UV-sampling
    fwidth = 0.5;
    fheight= 0.5;
    if strcmp(format,'400')
        fwidth = 0;
        fheight= 0;
    elseif strcmp(format,'411')
        fwidth = 0.25;
        fheight= 1;
    elseif strcmp(format,'yuv420p')
        fwidth = 0.5;
        fheight= 0.5;
    elseif strcmp(format,'yuv420p10le')
        fwidth = 0.5;
        fheight= 0.5;
    elseif strcmp(format,'422')
        fwidth = 0.5;
        fheight= 1;
    elseif strcmp(format,'444')
        fwidth = 1;
        fheight= 1;
    else
        display('Error: wrong format');
    end
    %get Filesize and Framenumber
    filep = dir(File); 
    fileBytes = filep.bytes; %Filesize
    clear filep
    framenumber = fileBytes/(width*height*(1+2*fheight*fwidth)); %Framenumber
    if strcmp(format,'yuv420p10le')
        framenumber = framenumber/2;
    end
    mov = [];
    if mod(framenumber,1) ~= 0
        display('Error: wrong resolution, format or filesize');
    else
        h = waitbar(0,'Please wait ... ');
        for cntf = 1:subsample:framenumber %read YUV-Frames
            waitbar(cntf/framenumber,h);
            YUV = loadFileYUV(width,height,cntf,File,fheight,fwidth,format);
            idx = ceil(cntf/subsample);
            mov(idx).cdata = YUV;
            mov(idx).colormap = [];
        end
        close(h);
    end



% loadFileYUV(width,height,Frame_Number,File,fheight,fwidth)
function PVD = loadFileYUV(width,heigth,Frame,fileName,Teil_h,Teil_b,format)
    % get size of U and V
    fileId = fopen(fileName,'r');
    width_h = width*Teil_b;
    heigth_h = heigth*Teil_h;
    % compute factor for framesize
    factor = 1+(Teil_h*Teil_b)*2;
    % compute framesize
    framesize = width*heigth;
      
    if strcmp(format,'yuv420p10le')
        fseek(fileId,2*(Frame-1)*factor*framesize, 'bof');
        YMatrix = fread(fileId, width * heigth, 'uint16');  % create Y-Matrix
        YMatrix = YMatrix / 4.0;
        YMatrix = int16(reshape(YMatrix,width,heigth)');
        if Teil_h == 0                                      % create U- and V- Matrix
            UMatrix = 0;
            VMatrix = 0;
        else
            UMatrix = fread(fileId,width_h * heigth_h, 'uint16')/4.0;
            UMatrix = int16(UMatrix);
            UMatrix = reshape(UMatrix,width_h, heigth_h).';
            VMatrix = fread(fileId,width_h * heigth_h, 'uint16')/4.0;
            VMatrix = int16(VMatrix);
            VMatrix = reshape(VMatrix,width_h, heigth_h).';       
        end
        YUV(1:heigth,1:width,1) = YMatrix;                  % compose the YUV-matrix:
    else
        fseek(fileId,(Frame-1)*factor*framesize, 'bof');
        YMatrix = fread(fileId, width * heigth, 'uchar');	% create Y-Matrix
        YMatrix = int16(reshape(YMatrix,width,heigth)');
        if Teil_h == 0                                      % create U- and V- Matrix
            UMatrix = 0;
            VMatrix = 0;
        else
            UMatrix = fread(fileId,width_h * heigth_h, 'uchar');
            UMatrix = int16(UMatrix);
            UMatrix = reshape(UMatrix,width_h, heigth_h).';
            VMatrix = fread(fileId,width_h * heigth_h, 'uchar');
            VMatrix = int16(VMatrix);
            VMatrix = reshape(VMatrix,width_h, heigth_h).';       
        end
        YUV(1:heigth,1:width,1) = YMatrix;                  % compose the YUV-matrix:
    end
    YUV(1:heigth,1:width,1) = YMatrix;                  % compose the YUV-matrix:
    if Teil_h == 0
        YUV(:,:,2) = 127;
        YUV(:,:,3) = 127;
    end
    % consideration of the subsampling of U and V
    if Teil_b == 1
        UMatrix1(:,:) = UMatrix(:,:);
        VMatrix1(:,:) = VMatrix(:,:);
    
    elseif Teil_b == 0.5        
        UMatrix1(1:heigth_h,1:width) = int16(0);
        UMatrix1(1:heigth_h,1:2:end) = UMatrix(:,1:1:end);
        UMatrix1(1:heigth_h,2:2:end) = UMatrix(:,1:1:end);
 
        VMatrix1(1:heigth_h,1:width) = int16(0);
        VMatrix1(1:heigth_h,1:2:end) = VMatrix(:,1:1:end);
        VMatrix1(1:heigth_h,2:2:end) = VMatrix(:,1:1:end);
    
    elseif Teil_b == 0.25
        UMatrix1(1:heigth_h,1:width) = int16(0);
        UMatrix1(1:heigth_h,1:4:end) = UMatrix(:,1:1:end);
        UMatrix1(1:heigth_h,2:4:end) = UMatrix(:,1:1:end);
        UMatrix1(1:heigth_h,3:4:end) = UMatrix(:,1:1:end);
        UMatrix1(1:heigth_h,4:4:end) = UMatrix(:,1:1:end);
        
        VMatrix1(1:heigth_h,1:width) = int16(0);
        VMatrix1(1:heigth_h,1:4:end) = VMatrix(:,1:1:end);
        VMatrix1(1:heigth_h,2:4:end) = VMatrix(:,1:1:end);
        VMatrix1(1:heigth_h,3:4:end) = VMatrix(:,1:1:end);
        VMatrix1(1:heigth_h,4:4:end) = VMatrix(:,1:1:end);
    end
    
    if Teil_h == 1
        YUV(:,:,2) = UMatrix1(:,:);
        YUV(:,:,3) = VMatrix1(:,:);
        
    elseif Teil_h == 0.5        
        YUV(1:heigth,1:width,2) = int16(0);
        YUV(1:2:end,:,2) = UMatrix1(:,:);
        YUV(2:2:end,:,2) = UMatrix1(:,:);
        
        YUV(1:heigth,1:width,3) = int16(0);
        YUV(1:2:end,:,3) = VMatrix1(:,:);
        YUV(2:2:end,:,3) = VMatrix1(:,:);
        
    elseif Teil_h == 0.25
        YUV(1:heigth,1:width,2) = int16(0);
        YUV(1:4:end,:,2) = UMatrix1(:,:);
        YUV(2:4:end,:,2) = UMatrix1(:,:);
        YUV(3:4:end,:,2) = UMatrix1(:,:);
        YUV(4:4:end,:,2) = UMatrix1(:,:);
        
        YUV(1:heigth,1:width) = int16(0);
        YUV(1:4:end,:,3) = VMatrix1(:,:);
        YUV(2:4:end,:,3) = VMatrix1(:,:);
        YUV(3:4:end,:,3) = VMatrix1(:,:);
        YUV(4:4:end,:,3) = VMatrix1(:,:);
    end
    
    PVD = uint8(YUV);
    
    fclose(fileId);
       